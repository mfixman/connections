from __future__ import annotations
from typing import Any

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping

import torch
from torch import nn

from .data import AxiomTrainingExample
from .dataset import NoParseableProblemsError, collect_problems_parallel, load_axiom_dataset
from .graph import collate_axiom_graphs
from .logs import log
from .metrics import prediction_metrics
from .model import (
    AxiomModelConfig,
    AxiomPredictionNetwork,
    AxiomPredictor,
    resolve_device,
    save_checkpoint,
    write_durably,
)
from .parallel import determine_worker_count
from .split import SPLIT_SCHEME, ProblemSplit
from .tptp import DEFAULT_STEP_LIMIT, DEFAULT_TIMEOUT_SECONDS
from .wandb_tracking import WandbConfig, WandbTracker


@dataclass(frozen=True, slots=True)
class AxiomTrainingConfig:
    epochs: int = 200
    batch_size: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    message_rounds: int = 3
    num_hidden_layers: int = 2
    activation: str = "relu"
    seed: int = 0
    device: str = "auto"
    step_limit: int = DEFAULT_STEP_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    sat_policy: str = "satresetcop"
    num_workers: int | None = None
    log_every: int = 10


PROGRESS_SECONDS = 60.0


def collect_examples(
    problems: list[str],
    *,
    tptp_root: str | Path | None,
    config: AxiomTrainingConfig,
    progress: Callable[[int, int, int, int, str, str], None] | None = None,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]]]:
    exs: list[AxiomTrainingExample] = []
    skipped: list[dict[str, str]] = []
    unparseable = 0
    workers = determine_worker_count(len(problems), config.num_workers)
    log(
        f"axiom-predictor: using {workers} worker "
        f"{'process' if workers == 1 else 'processes'} for proof collection"
    )
    for result in collect_problems_parallel(
        problems,
        tptp_root=tptp_root,
        step_limit=config.step_limit,
        timeout_seconds=config.timeout_seconds,
        sat_policy=config.sat_policy,
        num_workers=workers,
    ):
        problem = result.problem
        example = result.example
        outcome = result.outcome
        unparseable += int(not result.parseable)

        if example is None:
            skipped.append({"problem": problem, "outcome": outcome})
            log(f"skipping {problem}: {outcome}")
        else:
            exs.append(example)
            log(
                f"proved {problem}: {int(sum(example.labels))}/{len(example.labels)} axiom clauses in SAT core"
            )

        if progress is not None:
            progress(
                len(exs) + len(skipped),
                len(problems),
                len(exs),
                len(skipped),
                problem,
                outcome,
            )

    if not exs:
        if problems and unparseable == len(problems):
            raise NoParseableProblemsError(
                f"none of the {len(problems)} .p files could be parsed as FOF or CNF"
            )
        raise RuntimeError(
            "none of the supplied problems produced a usable SAT core; no training examples"
        )

    return exs, skipped


def train_axiom_predictor(
    problems: list[str] | tuple[str, ...] | None = None,
    *,
    dataset: str | Path | None = None,
    output_dir: str | Path,
    tptp_root: str | Path | None = None,
    config: AxiomTrainingConfig = AxiomTrainingConfig(),
    wandb_config: WandbConfig = WandbConfig(),
    run_properties: Mapping[str, object] | None = None,
    split: ProblemSplit = ProblemSplit(),
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")

    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    problem_list = list(problems or [])
    if dataset is not None and problem_list:
        raise ValueError("dataset cannot be combined with problem paths")
    if dataset is None and not problem_list:
        raise ValueError("training requires problem paths or a dataset")

    dataset_metadata: Mapping[str, object] | None = None
    dataset_examples: list[AxiomTrainingExample] | None = None
    dataset_skipped: list[dict[str, str]] = []
    if dataset is not None:
        _log(f"loading dataset from {dataset}")
        started = time.monotonic()
        dataset_examples, dataset_skipped, dataset_metadata = load_axiom_dataset(dataset)
        if not split.is_everything:
            total = len(dataset_examples)
            dataset_examples = [
                example for example in dataset_examples if split.contains(example.problem_path)
            ]
            dataset_skipped = [
                item for item in dataset_skipped if split.contains(item.get("problem", ""))
            ]
            _log(f"kept {len(dataset_examples)} of {total} examples in {split.describe()}")
            if not dataset_examples:
                raise RuntimeError(f"no training examples in {split.describe()} of {dataset}")
        problem_list = [example.problem_path for example in dataset_examples]
        shards = dataset_metadata.get("shards", [])
        _log(
            f"loaded {len(dataset_examples)} examples and {len(dataset_skipped)} failure records from {len(shards) or 1} {'file' if len(shards) <= 1 else 'shards'} in {time.monotonic() - started:.1f}s; collection settings {json.dumps(dataset_metadata.get('collection', {}), sort_keys=True)}"
        )
        _emit(
            "dataset_loaded",
            dataset=str(dataset),
            shards=shards,
            collection=dataset_metadata.get("collection", {}),
            examples=len(dataset_examples),
            failures=len(dataset_skipped),
        )

    collection = dataset_metadata.get("collection", {}) if dataset_metadata is not None else {}
    effective_sat_policy = (
        collection.get("sat_policy", config.sat_policy)
        if isinstance(collection, Mapping)
        else config.sat_policy
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config)
    config_payload["sat_policy"] = effective_sat_policy
    config_payload["split"] = split.to_dict()
    tracker = WandbTracker.start(
        wandb_config,
        job_type="train",
        run_config={
            **config_payload,
            "resolved_device": str(device),
            "problems": problem_list,
            "problems_requested": len(problem_list),
            "dataset": None if dataset is None else str(dataset),
            "cli_arguments": dict(run_properties or {}),
        },
        output_dir=output,
    )
    try:
        if dataset_examples is None:
            examples, skipped = collect_examples(
                problem_list,
                tptp_root=tptp_root,
                config=config,
                progress=None if tracker is None else tracker.log_collection,
            )
        else:
            examples, skipped = dataset_examples, dataset_skipped
        if tracker is not None:
            tracker.run.summary.update(
                {
                    "collection/problems_proved": len(examples),
                    "collection/problems_skipped": len(skipped),
                }
            )
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
            tracker = None

        raise

    _report_dataset(_dataset_summary(examples, skipped, batch_size=config.batch_size))
    training_workers = determine_worker_count(None, config.num_workers)
    torch.set_num_threads(training_workers)
    log(
        f"axiom-predictor: using {training_workers} Torch CPU "
        f"{'thread' if training_workers == 1 else 'threads'}"
    )

    model = AxiomPredictionNetwork(
        AxiomModelConfig(
            hidden_dim=config.hidden_dim,
            message_rounds=config.message_rounds,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
        )
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    _log(
        f"model: {parameters} parameters, hidden_dim {config.hidden_dim}, {config.message_rounds} message rounds, {config.num_hidden_layers} hidden layers, {config.activation}; device {device}; {config.epochs} epochs, batch size {config.batch_size}, learning rate {config.learning_rate:g}"
    )
    _emit("model", parameters=parameters, device=str(device), config=config_payload)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    rnd = random.Random(config.seed)
    batches = math.ceil(len(examples) / config.batch_size)
    _write_json(output / "training_config.json", config_payload)
    try:
        probabilities: list[float] = []
        epoch_metrics: dict[str, float | int | None] | None = None
        for epoch in range(1, config.epochs + 1):
            started = time.monotonic()
            last_progress = started
            model.train()
            order = list(examples)
            rnd.shuffle(order)
            total_loss = 0.0
            total_labels = 0
            norms: list[float] = []
            if epoch == 1:
                _log(f"epoch 1/{config.epochs}: starting {batches} batches")
            for index, chunk in enumerate(_chunks(order, config.batch_size), start=1):
                batch = collate_axiom_graphs([(example.graph, example.labels) for example in chunk])
                labels = batch.labels
                if labels is None:
                    raise RuntimeError("training batch unexpectedly has no labels")

                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = loss_fn(logits, labels)
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                optimizer.step()
                batch_loss = float(loss.detach().cpu().item())
                total_loss += batch_loss * len(labels)
                total_labels += len(labels)
                now = time.monotonic()
                if index < batches and (
                    (epoch == 1 and index == 1) or now - last_progress >= PROGRESS_SECONDS
                ):
                    last_progress = now
                    _report_progress(
                        epoch,
                        config.epochs,
                        index,
                        batches,
                        loss=batch_loss,
                        seconds=now - started,
                        labels=len(labels),
                    )
                    if tracker is not None:
                        tracker.log_progress(
                            epoch - 1 + index / batches,
                            loss=batch_loss,
                            batches_done=index,
                            batches_total=batches,
                        )

            loss_value = total_loss / total_labels
            train_seconds = time.monotonic() - started
            report = (
                epoch == 1
                or epoch == config.epochs
                or (config.log_every > 0 and epoch % config.log_every == 0)
            )
            epoch_metrics = None
            eval_seconds = None
            if report:
                evaluated = time.monotonic()
                _, probabilities, epoch_metrics = _example_outputs(
                    model, examples, batch_size=config.batch_size
                )
                eval_seconds = time.monotonic() - evaluated
            if report or not math.isfinite(loss_value):
                _report_epoch(
                    epoch,
                    config.epochs,
                    loss=loss_value,
                    seconds=train_seconds,
                    eval_seconds=eval_seconds,
                    norms=norms,
                    metrics=epoch_metrics,
                )
            if tracker is not None:
                tracker.log_epoch(
                    epoch,
                    loss=loss_value,
                    seconds=train_seconds,
                    grad_norm=sum(norms) / len(norms) if norms else None,
                    metrics=epoch_metrics,
                )
            if report and epoch < config.epochs:
                save_checkpoint(
                    output / "model.pt",
                    model,
                    training_config=config_payload,
                    epoch=epoch,
                )
                _log(f"saved epoch {epoch} checkpoint to {output / 'model.pt'}")

        if epoch_metrics is None:
            _, probabilities, epoch_metrics = _example_outputs(
                model, examples, batch_size=config.batch_size
            )
        metrics: dict[str, Any] = dict(epoch_metrics)
        metrics.update(
            {
                "evaluation_kind": "training-set overfit (not held-out generalization)",
                "label_semantics": "native CaDiCaL failed-assumption SAT-core membership",
                "sat_policy": effective_sat_policy,
                "dataset": None if dataset is None else str(dataset),
                "problems_proved": len(examples),
                "problems_skipped": len(skipped),
                "skipped": skipped,
            }
        )
        save_checkpoint(
            output / "model.pt",
            model,
            training_config=config_payload,
            epoch=config.epochs,
        )
        _write_json(output / "metrics.json", metrics)
        _log(f"saved checkpoint, config and metrics to {output}")
        if tracker is not None:
            tracker.log_results(examples, probabilities, metrics)
            tracker.log_model(output / "model.pt")

        return metrics
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
            tracker = None

        raise
    finally:
        if tracker is not None:
            tracker.finish()


def evaluate_axiom_predictor(
    checkpoint: str | Path,
    problems: list[str] | tuple[str, ...] | None = None,
    *,
    dataset: str | Path | None = None,
    tptp_root: str | Path | None = None,
    device: str = "auto",
    config: AxiomTrainingConfig | None = None,
    wandb_config: WandbConfig = WandbConfig(),
    run_properties: Mapping[str, object] | None = None,
    split: ProblemSplit = ProblemSplit(),
) -> dict[str, Any]:
    problem_list = list(problems or [])
    if dataset is not None and problem_list:
        raise ValueError("evaluate takes either PROBLEM inputs or --data-dir, not both")
    if dataset is None and not problem_list:
        raise ValueError("evaluation requires problem paths or a dataset")
    predictor = AxiomPredictor.load(checkpoint, device=device)
    _warn_on_training_overlap(predictor.training_config, split)
    collection_config = config or AxiomTrainingConfig(device=device)
    sat_policy = collection_config.sat_policy
    dataset_examples: list[AxiomTrainingExample] | None = None
    dataset_skipped: list[dict[str, str]] = []
    if dataset is not None:
        _log(f"loading dataset from {dataset}")
        dataset_examples, dataset_skipped, metadata = load_axiom_dataset(dataset)
        total = len(dataset_examples)
        dataset_examples = [
            example for example in dataset_examples if split.contains(example.problem_path)
        ]
        dataset_skipped = [
            item for item in dataset_skipped if split.contains(item.get("problem", ""))
        ]
        _log(
            f"evaluating {len(dataset_examples)} of {total} dataset examples ({split.describe()}) without proof search"
        )
        if not dataset_examples:
            raise RuntimeError(f"no examples in {split.describe()} of {dataset}")
        collection = metadata.get("collection", {})
        if isinstance(collection, Mapping):
            sat_policy = str(collection.get("sat_policy", sat_policy))
        problem_list = [example.problem_path for example in dataset_examples]
    tracker = WandbTracker.start(
        wandb_config,
        job_type="evaluate",
        run_config={
            "checkpoint": str(checkpoint),
            "device": str(predictor.device),
            "sat_policy": sat_policy,
            "dataset": None if dataset is None else str(dataset),
            "split": split.to_dict(),
            "training_split": predictor.training_config.get("split"),
            "problems": problem_list,
            "cli_arguments": dict(run_properties or {}),
        },
    )
    try:
        if dataset_examples is None:
            examples, skipped = collect_examples(
                problem_list,
                tptp_root=tptp_root,
                config=collection_config,
                progress=None if tracker is None else tracker.log_collection,
            )
        else:
            examples, skipped = dataset_examples, dataset_skipped
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise
    _report_dataset(_dataset_summary(examples, skipped, batch_size=collection_config.batch_size))
    started = time.monotonic()
    _, probabilities, raw_metrics = _example_outputs(
        predictor.model, examples, batch_size=collection_config.batch_size
    )
    metrics: dict[str, Any] = dict(raw_metrics)
    _log(
        f"evaluated {len(examples)} problems in {time.monotonic() - started:.1f}s: {_metric_text(metrics)}"
    )
    metrics.update(
        {
            "evaluation_kind": "labelled SAT-core-membership evaluation",
            "label_semantics": "native CaDiCaL failed-assumption SAT-core membership",
            "sat_policy": sat_policy,
            "dataset": None if dataset is None else str(dataset),
            "split": split.to_dict(),
            "training_split": predictor.training_config.get("split"),
            "problems_proved": len(examples),
            "problems_skipped": len(skipped),
            "skipped": skipped,
        }
    )
    if tracker is not None:
        try:
            tracker.log_results(examples, probabilities, metrics)
        finally:
            tracker.finish()

    return metrics


def _warn_on_training_overlap(training_config: Mapping[str, Any], split: ProblemSplit):
    trained = training_config.get("split")
    if not isinstance(trained, Mapping):
        _log(
            "warning: the checkpoint does not record its training split, so held-out evaluation cannot be checked"
        )
        return
    trained_split = ProblemSplit(
        int(trained.get("split", 1)),
        tuple(int(p) for p in trained.get("parts", (0,))),
        str(trained.get("by", "family")),
    )
    if not trained_split.is_everything and trained.get("scheme") != SPLIT_SCHEME:
        _log(
            f"warning: the checkpoint was trained on {trained_split.describe()} with an older split hash; its parts are not the parts this version selects, so problems may overlap"
        )
    elif trained_split.is_everything:
        _log(
            "warning: the checkpoint was trained on every problem of its dataset; metrics on problems from that dataset are not held-out"
        )
    elif (trained_split.split, trained_split.by) != (split.split, split.by):
        _log(
            f"warning: the checkpoint was trained on {trained_split.describe()}, a different split from {split.describe()}; problems may overlap"
        )
    elif set(trained_split.parts) & set(split.parts):
        _log(
            f"warning: evaluating {split.describe()} overlaps the training {trained_split.describe()}; metrics are not held-out"
        )


def evaluate_examples(
    model: AxiomPredictionNetwork,
    examples: list[AxiomTrainingExample],
    *,
    device: torch.device,
) -> dict[str, float | int | None]:
    _ = device
    return _example_outputs(model, examples, batch_size=1000)[2]


def _example_outputs(
    model: AxiomPredictionNetwork,
    examples: list[AxiomTrainingExample],
    *,
    batch_size: int,
) -> tuple[list[int], list[float], dict[str, float | int | None]]:
    model.eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(examples, batch_size):
            batch = collate_axiom_graphs([(example.graph, example.labels) for example in chunk])
            probabilities.extend(torch.sigmoid(model(batch)).cpu().tolist())

    labels = [int(label) for example in examples for label in example.labels]
    metrics = prediction_metrics(
        labels,
        probabilities,
        problem_sizes=[len(example.labels) for example in examples],
    )
    return labels, probabilities, metrics


def _log(message: str):
    log(f"axiom-predictor: {message}")


def _emit(event: str, **fields: object):
    print(json.dumps({"event": event, **fields}, sort_keys=True, default=str), flush=True)


def _dataset_summary(
    examples: list[AxiomTrainingExample],
    skipped: list[dict[str, str]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    sizes = [len(example.labels) for example in examples]
    cores = [int(sum(example.labels)) for example in examples]
    nodes = [sum(len(rows) for rows in example.graph.graph.nodes.values()) for example in examples]
    edges = [sum(len(rows) for rows in example.graph.graph.edges.values()) for example in examples]
    labels = sum(sizes)
    largest = sorted(zip(examples, sizes, nodes), key=lambda item: -item[1])[:5]
    return {
        "problems": len(examples),
        "duplicate_problems": len(examples) - len({example.problem_path for example in examples}),
        "problems_skipped": len(skipped),
        "skipped_outcomes": dict(
            Counter(_outcome_kind(item.get("outcome", "")) for item in skipped).most_common()
        ),
        "batches_per_epoch": math.ceil(len(examples) / batch_size),
        "axiom_clauses": labels,
        "core_clauses": sum(cores),
        "positive_rate": sum(cores) / labels if labels else None,
        "axioms_per_problem": _spread(sizes),
        "core_per_problem": _spread(cores),
        "core_fraction_per_problem": _spread([c / n for c, n in zip(cores, sizes) if n]),
        "problems_without_axioms": sum(n == 0 for n in sizes),
        "problems_with_empty_core": sum(c == 0 for c in cores),
        "problems_with_full_core": sum(c == n and n > 0 for c, n in zip(cores, sizes)),
        "graph_nodes_per_problem": _spread(nodes),
        "graph_edges_per_problem": _spread(edges),
        "largest_problems": [
            {"problem": example.problem_path, "axiom_clauses": n, "graph_nodes": g}
            for example, n, g in largest
        ],
    }


def _report_dataset(summary: dict[str, Any]):
    rate = summary["positive_rate"]
    _log(
        f"dataset: {summary['problems']} problems ({summary['problems_skipped']} skipped, {summary['duplicate_problems']} duplicates), {summary['axiom_clauses']} axiom clauses, {summary['core_clauses']} in SAT cores ({'n/a' if rate is None else f'{100 * rate:.2f}%'}), {summary['batches_per_epoch']} batches per epoch"
    )
    if summary["skipped_outcomes"]:
        _log(
            "skipped outcomes: "
            + ", ".join(f"{kind} x{count}" for kind, count in summary["skipped_outcomes"].items())
        )
    _log(f"axiom clauses per problem: {_spread_text(summary['axioms_per_problem'])}")
    _log(
        f"SAT-core clauses per problem: {_spread_text(summary['core_per_problem'])}; core fraction {_spread_text(summary['core_fraction_per_problem'])}"
    )
    _log(
        f"{summary['problems_with_empty_core']} problems with an empty axiom core, {summary['problems_with_full_core']} with every axiom in the core, {summary['problems_without_axioms']} without axioms"
    )
    _log(
        f"graph nodes per problem: {_spread_text(summary['graph_nodes_per_problem'])}; edges per problem: {_spread_text(summary['graph_edges_per_problem'])}"
    )
    _log(
        "largest problems: "
        + ", ".join(
            f"{item['problem']} ({item['axiom_clauses']} axioms, {item['graph_nodes']} nodes)"
            for item in summary["largest_problems"]
        )
    )
    _emit("dataset", **summary)


def _report_progress(
    epoch: int,
    epochs: int,
    index: int,
    batches: int,
    *,
    loss: float,
    seconds: float,
    labels: int,
):
    _log(
        f"epoch {epoch}/{epochs}: batch {index}/{batches} done after {seconds:.1f}s, batch loss {loss:.6f} over {labels} axiom clauses"
    )
    _emit(
        "progress",
        epoch=epoch,
        epochs=epochs,
        batch=index,
        batches=batches,
        loss=loss,
        seconds=seconds,
        labels=labels,
    )


def _report_epoch(
    epoch: int,
    epochs: int,
    *,
    loss: float,
    seconds: float,
    eval_seconds: float | None,
    norms: list[float],
    metrics: dict[str, float | int | None] | None,
):
    norm = sum(norms) / len(norms) if norms else None
    warning = "" if math.isfinite(loss) else " WARNING: non-finite loss"
    suffix = (
        ""
        if metrics is None
        else f"; train-set {_metric_text(metrics)} (evaluated in {eval_seconds:.2f}s)"
    )
    _log(
        f"epoch {epoch}/{epochs}: loss {loss:.6f}, mean grad norm {'n/a' if norm is None else f'{norm:.4f}'}, {len(norms)} batches in {seconds:.2f}s{suffix}{warning}"
    )
    _emit(
        "epoch",
        epoch=epoch,
        epochs=epochs,
        loss=loss,
        seconds=seconds,
        eval_seconds=eval_seconds,
        batches=len(norms),
        grad_norm_mean=norm,
        grad_norm_max=max(norms, default=None),
        metrics=metrics,
    )


def _metric_text(metrics: Mapping[str, object]) -> str:
    names = (
        "bce",
        "roc_auc",
        "average_precision",
        "f1_at_0.5",
        "macro_recall_at_1",
        "macro_recall_at_5",
        "macro_recall_at_10",
    )
    return ", ".join(
        f"{name} {'n/a' if metrics.get(name) is None else f'{metrics[name]:.4f}'}"
        for name in names
        if name in metrics
    )


def _spread(values: list[float] | list[int]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _spread_text(spread: object) -> str:
    if not isinstance(spread, Mapping):
        return "n/a"
    return ", ".join(f"{name} {value:.3g}" for name, value in spread.items())


def _outcome_kind(outcome: str) -> str:
    head, _, rest = outcome.partition(": ")
    code = re.match(r"\s*(\[[A-Z0-9_]+\])", rest)
    return head if code is None else f"{head} {code.group(1)}"


def _chunks(examples: list[AxiomTrainingExample], size: int):
    if size < 1:
        raise ValueError("batch size must be at least 1")

    for start in range(0, len(examples), size):
        yield examples[start : start + size]


def _write_json(path: Path, value: object):
    write_durably(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


__all__ = [
    "AxiomTrainingConfig",
    "collect_examples",
    "evaluate_axiom_predictor",
    "evaluate_examples",
    "train_axiom_predictor",
]
