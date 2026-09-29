from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any

from .data import AxiomTrainingExample
from .logs import log
from .metrics import prediction_metrics

DEFAULT_WANDB_ENTITY = "mfixman-phd-team"
DEFAULT_WANDB_PROJECT = "axiom-prediction"
SERVICE_WAIT_SECONDS = 300

@dataclass(frozen = True, slots = True)
class WandbConfig:
    enabled: bool | None = True
    entity: str = DEFAULT_WANDB_ENTITY
    project: str = DEFAULT_WANDB_PROJECT

    name: str | None = None
    key_file: Path = Path("secrets/wandb_key")

    name_prefix: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

class WandbTracker:
    def __init__(self, module: Any, run: Any):
        self.wandb = module
        self.run = run
        self.best_evaluation = None

    @classmethod
    def start(
        cls,
        config: WandbConfig,
        *,
        job_type: str,
        run_config: dict[str, Any],
        output_dir: Path | None = None,
    ) -> "WandbTracker | None":
        if config.enabled is False:
            return None

        key = os.environ.get("WANDB_API_KEY", "").strip() or None
        key_path = config.key_file.expanduser()
        if key is None and not key_path.is_file():
            log(
                "warning: W&B disabled because neither WANDB_API_KEY nor "
                f"{key_path} is available; use --no-wandb to suppress this warning"
            )

            return None

        # wandb-core can take longer than the default 30s to start when it is
        # read from a slow shared filesystem, e.g. on a busy HPC node.

        os.environ.setdefault("WANDB__SERVICE_WAIT", str(SERVICE_WAIT_SECONDS))

        try:
            wandb = importlib.import_module("wandb")
        except ImportError:
            log("warning: W&B disabled because wandb is not installed; install with python -m pip install wandb")
            return None

        file_key = None
        if key is None:
            try:
                file_key = key_path.read_text(encoding = "utf-8").strip()
            except (OSError, UnicodeError):
                log(f"warning: W&B disabled because its key file cannot be read: {key_path}")
                return None

            if not file_key:
                log(f"warning: W&B disabled because its key file is empty: {key_path}")
                return None

        try:
            if file_key is not None:
                wandb.login(key = file_key)

            run = wandb.init(
                entity = config.entity,
                project = config.project,
                name = config.name,
                group = config.group,
                tags = list(config.tags) or None,

                job_type = job_type,
                config = run_config,
                dir = None if output_dir is None else str(output_dir),
            )

            if run is None:
                raise RuntimeError("wandb.init() did not create a run")
        except Exception as error:
            log(
                f"warning: W&B disabled because it failed to start ({error}); "
                "continuing without tracking"
            )

            return None

        name = getattr(run, "name", None)
        if config.name_prefix and name and not name.startswith(config.name_prefix):
            try:
                run.name = f"{config.name_prefix}-{name}"
            except Exception as error:
                log(
                    f"warning: could not rename W&B run {name!r} with prefix {config.name_prefix!r} ({error})"
                )

        if hasattr(run, "define_metric"):
            run.define_metric("epoch")
            run.define_metric("train/*", step_metric = "epoch")
            run.define_metric("evaluation/*", step_metric = "epoch")
            run.define_metric("progress/epoch")
            run.define_metric("progress/*", step_metric = "progress/epoch")

        return cls(wandb, run)

    def log_epoch(
        self,
        epoch: int,
        *,

        loss: float,
        seconds: float | None = None,
        grad_norm: float | None = None,
        metrics: dict[str, float | int | None] | None = None,
        evaluation: dict[str, float | int | None] | None = None,
    ):
        payload: dict[str, Any] = {
            "epoch": epoch,
            "train/loss": loss,
        }

        if seconds is not None:
            payload["train/epoch_seconds"] = seconds

        if grad_norm is not None:
            payload["train/grad_norm_mean"] = grad_norm

        payload.update(
            {
                f"train/{name}": value
                for name, value in (metrics or {}).items()
                if name not in {"examples", "positives"} and value is not None
            }
        )

        payload.update(
            {
                f"evaluation/{name}": value
                for name, value in (evaluation or {}).items()
                if value is not None
            }
        )

        self.run.log(payload)
        self.update_best_evaluation(epoch, evaluation)

    def update_best_evaluation(self, epoch, evaluation):
        score = (evaluation or {}).get("macro_average_precision")
        if score is None or (
            self.best_evaluation is not None and score <= self.best_evaluation
        ):
            return

        self.best_evaluation = score
        self.run.summary.update(
            {
                "evaluation/best_macro_average_precision": score,
                "evaluation/best_epoch": epoch,
            }
        )

    def log_evaluation_results(self, examples, probabilities, metrics):
        self.run.summary.update(
            {
                f"evaluation/{name}": value
                for name, value in metrics.items()
                if isinstance(value, int | float) or value is None
            }
        )

        self.run.log(
            {
                "evaluation_results/per_problem": self.problem_table(examples, probabilities),
                "evaluation_results/ranked_axioms": self.prediction_table(examples, probabilities),
            }
        )

    def log_progress(
        self,
        epoch: float,
        *,
        loss: float,
        batches_done: int,
        batches_total: int,
    ):
        self.run.log(
            {
                "progress/epoch": epoch,
                "progress/batch_loss": loss,
                "progress/batches_done": batches_done,
                "progress/batches_total": batches_total,
            }
        )

    def log_collection(
        self,
        processed: int,
        total: int,
        proved: int,

        skipped: int,
        problem: str,
        outcome: str,
    ):
        self.run.log(
            {
                "collection/problems_processed": processed,
                "collection/problems_total": total,
                "collection/fraction_complete": processed / total,

                "collection/problems_proved": proved,
                "collection/problems_skipped": skipped,

                "collection/current_problem": problem,
                "collection/current_outcome": outcome,
            }
        )

    def log_results(
        self,
        examples: list[AxiomTrainingExample],
        probabilities: list[float],
        metrics: dict[str, Any],
    ):
        labels = [int(label) for example in examples for label in example.labels]
        self.run.summary.update(
            {
                f"result/{name}": value
                for name, value in metrics.items()
                if isinstance(value, int | float) or value is None
            }
        )

        plots = {
            "results/axiom_probabilities": self.wandb.Histogram(
                probabilities,
                num_bins = 32,
            ),
            "results/ranked_axioms": self.prediction_table(examples, probabilities),
            "results/per_problem": self.problem_table(examples, probabilities),
        }

        used = [p for p, label in zip(probabilities, labels, strict = True) if label]
        unused = [
            p
            for p, label in zip(probabilities, labels, strict = True)
            if not label
        ]

        if used:
            plots["results/in_sat_core_axiom_probabilities"] = self.wandb.Histogram(
                used,
                num_bins = 32,
            )

        if unused:
            plots["results/not_in_sat_core_axiom_probabilities"] = self.wandb.Histogram(
                unused,
                num_bins = 32,
            )

        self.run.log(plots)
        if 0 < sum(labels) < len(labels):
            roc = self.wandb.Table(
                data = roc_points(labels, probabilities),
                columns = ["false_positive_rate", "true_positive_rate"],
            )

            precision_recall = self.wandb.Table(
                data = precision_recall_points(labels, probabilities),
                columns = ["recall", "precision"],
            )

            self.run.log(
                {
                    "results/roc_curve": self.wandb.plot.line(
                        roc,
                        "false_positive_rate",
                        "true_positive_rate",
                        title = "ROC curve",
                    ),
                    "results/precision_recall_curve": self.wandb.plot.line(
                        precision_recall,
                        "recall",
                        "precision",
                        title = "Precision-recall curve",
                    ),
                }
            )

    def log_run_result(
        self,
        processed: int,
        total: int,
        proved: int,
        result: dict[str, Any],
    ):
        self.run.log(
            {
                "run/problems_processed": processed,
                "run/problems_total": total,
                "run/fraction_complete": processed / total,
                "run/problems_proved": proved,
                "run/current_problem": str(result.get("problem")),
                "run/current_outcome": str(result.get("outcome")),
            }
        )

    def log_run_results(self, results: list[dict[str, Any]], summary: dict[str, Any]):
        self.run.summary.update(
            {
                f"run/{k}": v
                for k, v in summary.items()
                if isinstance(v, int | float) or v is None
            }
        )

        columns = [
            "problem",
            "outcome",
            "proved",

            "seconds",
            "steps",
            "proof_size",

            "prediction_seconds",
            "axioms",
            "kept_axioms",
        ]

        self.run.log(
            {
                "run/results": self.wandb.Table(
                    data = [[r.get(c) for c in columns] for r in results],
                    columns = columns,
                )
            }
        )

    def log_model(self, checkpoint: Path):
        self.run.log_model(path = str(checkpoint), name = "axiom-predictor")

    def finish(self, *, exit_code: int = 0):
        self.run.finish(exit_code = exit_code)

    def prediction_table(
        self,
        examples: list[AxiomTrainingExample],
        probabilities: list[float],
    ) -> Any:
        rows: list[list[object]] = []
        offset = 0
        for example in examples:
            size = len(example.labels)
            scores = probabilities[offset : offset + size]
            ranks = {
                local_index: rank
                for rank, local_index in enumerate( sorted(range(size), key = lambda index: (-scores[index], index)), start = 1, )
            }

            for local_index, (clause_index, label, probability) in enumerate(
                zip(example.graph.axiom_clause_ids, example.labels, scores, strict = True)
            ):
                if example.axiom_clause_texts:
                    clause_text = example.axiom_clause_texts[local_index]
                elif example.matrix is not None:
                    clause_text = str(example.matrix.clauses[clause_index])
                else:
                    clause_text = ""

                rows.append(
                    [
                        example.problem_path,
                        clause_index,
                        clause_text,
                        int(label),
                        probability,
                        ranks[local_index],
                    ]
                )

            offset += size

        rows.sort(key = lambda row: (str(row[0]), int(row[5])))
        return self.wandb.Table(
            data = rows[:10_000],
            columns = [
                "problem",
                "clause_index",
                "clause_text",
                "in_sat_core",
                "probability",
                "rank",
            ],
        )

    def problem_table(
        self,
        examples: list[AxiomTrainingExample],
        probabilities: list[float],
    ) -> Any:
        rows: list[list[object]] = []
        offset = 0
        for example in examples:
            size = len(example.labels)
            scores = probabilities[offset : offset + size]
            labels = [int(label) for label in example.labels]
            metrics = prediction_metrics(labels, scores, problem_sizes = [size])
            rows.append(
                [
                    example.problem_path,
                    size,
                    sum(labels),

                    metrics["bce"],
                    metrics["roc_auc"],
                    metrics["average_precision"],

                    metrics["macro_recall_at_1"],
                    metrics["macro_recall_at_3"],
                    metrics["macro_recall_at_5"],
                    metrics["macro_recall_at_10"],
                ]
            )

            offset += size

        return self.wandb.Table(
            data = rows,
            columns = [
                "problem",
                "axiom_clauses",
                "axioms_in_sat_core",

                "bce",
                "roc_auc",
                "average_precision",

                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
            ],
        )

def roc_points(labels: list[int], scores: list[float]) -> list[list[float]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    points = [[0.0, 0.0]]
    true_positives = 0
    false_positives = 0
    for _score, label in sorted(
        zip(scores, labels, strict = True),
        key = lambda item: item[0],
        reverse = True,
    ):
        true_positives += label
        false_positives += 1 - label
        points.append([false_positives / negatives, true_positives / positives])

    return points

def precision_recall_points(labels: list[int], scores: list[float]) -> list[list[float]]:
    positives = sum(labels)
    points = [[0.0, 1.0]]
    true_positives = 0
    for count, (_score, label) in enumerate(
        sorted(
            zip(scores, labels, strict = True),
            key = lambda item: item[0],
            reverse = True,
        ),
        start = 1,
    ):
        true_positives += label
        points.append([true_positives / positives, true_positives / count])

    return points

__all__ = [
    "DEFAULT_WANDB_ENTITY",
    "DEFAULT_WANDB_PROJECT",
    "WandbConfig",
    "WandbTracker",
]
