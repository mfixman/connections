from __future__ import annotations

from .choices import GraphInputKind, ProverPolicy, SplitKey, plain_values

from typing import Any

from dataclasses import asdict, dataclass, replace
from functools import partial
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
from .evaluation_resume import resume_collection, resume_predictions
from .logs import log
from .logs import progress as track_progress
from .output import write_record
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
from .models import load_model_class
from .split import SPLIT_SCHEME, ProblemSplit
from .tptp import DEFAULT_STEP_LIMIT, DEFAULT_TIMEOUT_SECONDS
from .wandb_tracking import WandbConfig, WandbTracker
from .resume import check_training_target, dataset_fingerprint, next_training_run, restore_training, snapshot_training

@dataclass(frozen = True, slots = True)
class AxiomTrainingConfig:
    epochs: int = 200
    batch_size: int = 43
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    hidden_dim: int = 64
    message_rounds: int = 3
    num_hidden_layers: int = 2
    activation: str = "relu"
    graph_input: str = GraphInputKind.Full
    model: str | None = None

    seed: int = 0
    device: str = "cuda"

    step_limit: int = DEFAULT_STEP_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    sat_policy: ProverPolicy | str | None = None

    num_workers: int | None = None
    log_every: int = 10

    def __post_init__(self):
        self.network_config()
        if self.sat_policy is not None:
            object.__setattr__(self, "sat_policy", ProverPolicy(self.sat_policy))

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")

    def network_config(self) -> AxiomModelConfig:
        if self.model is not None:
            return load_model_class(self.model).default_config

        return AxiomModelConfig(
            hidden_dim = self.hidden_dim,
            message_rounds = self.message_rounds,
            num_hidden_layers = self.num_hidden_layers,
            activation = self.activation,
            graph_input = self.graph_input,
        )

def label_policy(requested, metadata = None):
    collection = (metadata or {}).get("collection", {})
    recorded = collection.get("sat_policy") if isinstance(collection, Mapping) else None
    if requested is not None:
        requested = ProverPolicy(requested)

    if recorded is not None:
        recorded = ProverPolicy(recorded)

    if requested is not None and recorded is not None and requested != recorded:
        raise ValueError(
            f"dataset labels were collected with {recorded}, but --policy requests {requested}; "
            "collect a separate dataset to change the label policy"
        )

    return ProverPolicy(recorded or requested or ProverPolicy.SatResetCoP)

PROGRESS_SECONDS = 60.0

def collect_examples(
    problems: list[str],
    *,
    tptp_root: str | Path | None,
    config: AxiomTrainingConfig,
    progress: Callable[[int, int, int, int, str, str], None] | None = None,
    resume = None,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]]]:
    exs: list[AxiomTrainingExample] = []
    skipped: list[dict[str, str]] = []
    unparseable = 0
    workers = determine_worker_count(len(problems), config.num_workers)
    log(
        f"axiom-predictor: using {workers} worker "
        f"{'process' if workers == 1 else 'processes'} for proof collection"
    )

    results = resume_collection(
        problems,
        collect_problems_parallel,
        resume,
        tptp_root = tptp_root,
        step_limit = config.step_limit,
        timeout_seconds = config.timeout_seconds,
        sat_policy = label_policy(config.sat_policy),
        num_workers = workers,
    )

    for result in results:
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
    evaluation_split: ProblemSplit | None = None,
    evaluation_problems: list[str] | tuple[str, ...] | None = None,
    evaluate_every: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    check_training_target(output_dir, resume)
    if evaluate_every < 0:
        raise ValueError("evaluate_every must be nonnegative")

    if evaluation_split is not None and (
        (split.split, split.by) != (evaluation_split.split, evaluation_split.by)
        or set(split.parts) & set(evaluation_split.parts)
    ):
        raise ValueError("evaluation parts must use the training split and not overlap --parts")

    if evaluation_problems is not None and (evaluation_split is None or dataset is not None):
        raise ValueError("evaluation_problems requires evaluation_split and no dataset")

    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")

    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    problem_list = list(problems or [])
    if dataset is not None and problem_list:
        raise ValueError("dataset cannot be combined with problem paths")

    if dataset is None and not problem_list:
        raise ValueError("training requires problem paths or a dataset")

    if evaluation_split is not None and dataset is None:
        problem_list = list(split.select(problem_list))
        if not problem_list:
            raise ValueError("no training problems in the selected parts")

        evaluation_problems = evaluation_split.select(list(evaluation_problems or []))
        if not evaluation_problems:
            raise ValueError("no evaluation problems supplied for the held-out parts")

    dataset_metadata: Mapping[str, object] | None = None
    dataset_examples: list[AxiomTrainingExample] | None = None
    dataset_skipped: list[dict[str, str]] = []
    held_out: list[AxiomTrainingExample] = []
    held_out_skipped: list[dict[str, str]] = []
    if dataset is not None:
        log_message(f"loading dataset from {dataset}")
        started = time.monotonic()
        dataset_examples, dataset_skipped, dataset_metadata = load_axiom_dataset(dataset)
        if evaluation_split is not None:
            held_out = [
                example
                for example in dataset_examples
                if evaluation_split.contains(example.problem_path)
            ]

            held_out_skipped = [
                item
                for item in dataset_skipped
                if evaluation_split.contains(item.get("problem", ""))
            ]

            if not held_out:
                raise ValueError(f"no evaluation examples in {evaluation_split.describe()}")

        if not split.is_everything:
            total = len(dataset_examples)
            dataset_examples = [
                example
                for example in dataset_examples
                if split.contains( example.problem_path )
            ]

            dataset_skipped = [
                item
                for item in dataset_skipped
                if split.contains( item.get("problem", "") )
            ]

            log_message(f"kept {len(dataset_examples)} of {total} examples in {split.describe()}")
            if not dataset_examples:
                raise RuntimeError(f"no training examples in {split.describe()} of {dataset}")

        problem_list = [example.problem_path for example in dataset_examples]
        shards = dataset_metadata.get("shards", [])
        log_message(
            f"loaded {len(dataset_examples)} examples and {len(dataset_skipped)} failure records from {len(shards) or 1} {'file' if len(shards) <= 1 else 'shards'} in {time.monotonic() - started:.1f}s; collection settings {json.dumps(dataset_metadata.get('collection', {}), sort_keys = True)}"
        )

        emit(
            "dataset_loaded",
            dataset = str(dataset),
            shards = shards,
            collection = dataset_metadata.get("collection", {}),
            examples = len(dataset_examples),
            failures = len(dataset_skipped),
        )

    effective_sat_policy = label_policy(config.sat_policy, dataset_metadata)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    output = Path(output_dir)
    output.mkdir(parents = True, exist_ok = True)

    config_payload = plain_values(asdict(config))
    network_config = config.network_config()
    config_payload.update(network_config.to_dict())
    config_payload["sat_policy"] = effective_sat_policy.wire_value
    config_payload["split"] = split.to_dict()

    run_count = next_training_run(output, resume)
    run_name = config.model or AxiomPredictionNetwork.__name__
    if run_count > 1:
        run_name = f"{run_name}-{run_count}"

    tracker = WandbTracker.start(
        replace(wandb_config, name = wandb_config.name or run_name, name_prefix = None),
        job_type = "train",
        run_config = {
            **config_payload,
            "run_count": run_count,
            "resolved_device": str(device),
            "problems": problem_list,
            "problems_requested": len(problem_list),
            "dataset": None if dataset is None else str(dataset),
            "evaluation_split": None if evaluation_split is None else evaluation_split.to_dict(),
            "evaluate_every": evaluate_every,
            "cli_arguments": dict(run_properties or {}),
        },
        output_dir = output,
    )

    try:
        if dataset_examples is None:
            examples, skipped = collect_examples(
                problem_list,
                tptp_root = tptp_root,
                config = config,
                progress = None if tracker is None else tracker.log_collection,
            )
        else:
            examples, skipped = dataset_examples, dataset_skipped

        if evaluation_split is not None and dataset is None:
            held_out, held_out_skipped = collect_examples(
                list(evaluation_problems),
                tptp_root = tptp_root,
                config = config,
            )

        if tracker is not None:
            tracker.run.summary.update(
                {
                    "collection/problems_proved": len(examples),
                    "collection/problems_skipped": len(skipped),
                }
            )

        report_dataset(
            dataset_summary(examples, skipped, batch_size = config.batch_size)
        )

        training_workers = determine_worker_count(None, config.num_workers)
        torch.set_num_threads(training_workers)
        log(
            f"axiom-predictor: using {training_workers} Torch CPU "
            f"{'thread' if training_workers == 1 else 'threads'}"
        )

        model_class = (
            AxiomPredictionNetwork
            if config.model is None
            else load_model_class(config.model)
        )

        model = model_class(network_config).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        log_message(
            f"model: {parameters} parameters, hidden_dim {network_config.hidden_dim}, {network_config.message_rounds} message rounds, {network_config.num_hidden_layers} hidden layers, {network_config.activation}; device {device}; {config.epochs} epochs, batch size {config.batch_size}, learning rate {config.learning_rate:g}"
        )

        emit(
            "model",
            parameters = parameters,
            device = str(device),
            config = config_payload,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = config.learning_rate,
            weight_decay = config.weight_decay,
        )

        loss_fn = nn.BCEWithLogitsLoss()
        rnd = random.Random(config.seed)
        log_message(f"fingerprinting {len(examples)} training examples")
        fingerprint = dataset_fingerprint(examples)
        first_epoch = 1
        if resume:
            log_message(f"restoring optimizer and RNG state from {output / 'model.pt'}")
            first_epoch = restore_training(
                output / "model.pt",
                model,
                optimizer,
                rnd,

                config_payload,
                fingerprint,
            )

        batches = math.ceil(len(examples) / config.batch_size)
        write_json(output / "training_config.json", config_payload)

        probabilities: list[float] = []
        epoch_metrics: dict[str, float | int | None] | None = None
        evaluation_metrics = None
        evaluation_probabilities = []
        evaluation_seconds = 0.0
        training_since_evaluation = 0.0
        last_evaluation = None

        for epoch in range(first_epoch, config.epochs + 1):
            started = time.monotonic()
            last_progress = started
            model.train()

            order = list(examples)
            rnd.shuffle(order)

            total_loss = 0.0
            total_labels = 0
            norms: list[float] = []

            log_message(f"epoch {epoch}/{config.epochs}: starting {batches} batches")

            for index, chunk in enumerate(chunks(order, config.batch_size), start = 1):
                batch = collate_axiom_graphs(
                    [(example.graph, example.labels) for example in chunk]
                )

                labels = batch.labels
                if labels is None:
                    raise RuntimeError("training batch unexpectedly has no labels")

                labels = labels.to(device)
                optimizer.zero_grad()

                logits = model(batch)
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss; checkpoint unchanged")

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite = True,
                )

                norms.append(float(grad_norm))
                optimizer.step()

                batch_loss = float(loss.detach().cpu().item())
                total_loss += batch_loss * len(labels)
                total_labels += len(labels)

                now = time.monotonic()
                if index < batches and (
                    (epoch == 1 and index == 1) or now - last_progress >= PROGRESS_SECONDS
                ):
                    last_progress = now
                    report_progress(
                        epoch,
                        config.epochs,
                        index,
                        batches,

                        loss = batch_loss,
                        seconds = now - started,
                        labels = len(labels),
                    )

                    if tracker is not None:
                        tracker.log_progress(
                            epoch - 1 + index / batches,
                            loss = batch_loss,
                            batches_done = index,
                            batches_total = batches,
                        )

            loss_value = total_loss / total_labels
            train_seconds = time.monotonic() - started
            training_since_evaluation += train_seconds
            evaluation_due = held_out and (
                last_evaluation is None
                or epoch == config.epochs
                or (
                    epoch - last_evaluation >= evaluate_every
                    if evaluate_every
                    else training_since_evaluation >= 10 * evaluation_seconds
                )
            )

            current_evaluation = None
            if evaluation_due:
                log_message(f"epoch {epoch}: evaluating {len(held_out)} held-out problems")
                evaluated = time.monotonic()
                _, evaluation_probabilities, evaluation_metrics = example_outputs(
                    model,
                    held_out,
                    batch_size = config.batch_size,
                )

                evaluation_seconds = time.monotonic() - evaluated
                current_evaluation = {
                    **evaluation_metrics,
                    "seconds": evaluation_seconds,
                    "problems": len(held_out),
                    "problems_skipped": len(held_out_skipped),
                }

                last_evaluation = epoch
                training_since_evaluation = 0.0
                log_message(
                    f"epoch {epoch}: held-out {metric_text(evaluation_metrics)} "
                    f"({len(held_out)} problems in {evaluation_seconds:.2f}s)"
                )

                emit("evaluation", epoch = epoch, **current_evaluation)

            report = (
                epoch == 1
                or epoch == config.epochs
                or (config.log_every > 0 and epoch % config.log_every == 0)
            )

            epoch_metrics = None
            eval_seconds = None
            if report:
                log_message(f"epoch {epoch}: computing training-set metrics")
                evaluated = time.monotonic()
                _, probabilities, epoch_metrics = example_outputs(
                    model,
                    examples,
                    batch_size = config.batch_size,
                )

                eval_seconds = time.monotonic() - evaluated

            report_epoch(
                epoch,
                config.epochs,

                loss = loss_value,
                seconds = train_seconds,
                eval_seconds = eval_seconds,

                norms = norms,
                metrics = epoch_metrics,
            )

            if tracker is not None:
                tracker.log_epoch(
                    epoch,
                    loss = loss_value,
                    seconds = train_seconds,
                    grad_norm = sum(norms) / len(norms) if norms else None,
                    metrics = epoch_metrics,
                    evaluation = current_evaluation,
                )

            epoch_path = output / f"epoch-{epoch:04d}.pt"
            log_message(f"saving epoch {epoch} checkpoint to {epoch_path}")
            save_checkpoint(
                epoch_path,
                model,
                training_config = config_payload,
                epoch = epoch,
                training_state = snapshot_training(optimizer, rnd, fingerprint),
            )

            write_durably(output / "model.pt", epoch_path.read_bytes())
            log_message(f"saved epoch {epoch} checkpoint to {epoch_path}")

        if epoch_metrics is None:
            _, probabilities, epoch_metrics = example_outputs(
                model,
                examples,
                batch_size = config.batch_size,
            )

        metrics: dict[str, Any] = dict(epoch_metrics)
        metrics.update(
            {
                "evaluation_kind": "training-set overfit (not held-out generalization)",
                "label_semantics": "native CaDiCaL failed-assumption SAT-core membership",
                "sat_policy": effective_sat_policy.wire_value,
                "dataset": None if dataset is None else str(dataset),

                "problems_proved": len(examples),
                "problems_skipped": len(skipped),
                "skipped": skipped,
            }
        )

        write_json(output / "metrics.json", metrics)
        if evaluation_metrics is not None:
            evaluation_metrics = {
                **evaluation_metrics,
                "evaluation_kind": "held-out SAT-core-membership evaluation",
                "split": evaluation_split.to_dict(),
                "epoch": last_evaluation,
                "seconds": evaluation_seconds,
                "problems_proved": len(held_out),
                "problems_skipped": len(held_out_skipped),
                "skipped": held_out_skipped,
            }

            write_json(output / "evaluation_metrics.json", evaluation_metrics)

        log_message(f"saved checkpoint, config and metrics to {output}")
        if tracker is not None:
            tracker.log_results(examples, probabilities, metrics, split = split)
            if evaluation_metrics is not None:
                tracker.log_evaluation_results(
                    held_out,
                    evaluation_probabilities,
                    evaluation_metrics,
                    split = evaluation_split,
                )

            tracker.log_model(output / "model.pt")

        return metrics
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code = 1)
            tracker = None

        raise
    finally:
        if tracker is not None:
            tracker.finish()

def evaluate_axiom_predictor(
    checkpoint: str | Path,
    problems: list[str] | tuple[str, ...] | None = None,
    *,
    multiprocess: bool = False,
    resume = None,
    dataset: str | Path | None = None,
    tptp_root: str | Path | None = None,
    device: str = "cuda",

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

    predictor = AxiomPredictor.load(checkpoint, device = device)
    warn_on_training_overlap(predictor.training_config, split)

    collection_config = config or AxiomTrainingConfig(device = device)
    checkpoint_policy = label_policy(
        collection_config.sat_policy,
        {"collection": predictor.training_config},
    )

    collection_config = replace(collection_config, sat_policy = checkpoint_policy)
    torch.set_num_threads(determine_worker_count(None, collection_config.num_workers))
    sat_policy = label_policy(collection_config.sat_policy)

    dataset_examples: list[AxiomTrainingExample] | None = None
    dataset_skipped: list[dict[str, str]] = []
    if dataset is not None:
        log_message(f"loading dataset from {dataset}")
        dataset_examples, dataset_skipped, metadata = load_axiom_dataset(dataset)
        total = len(dataset_examples)
        dataset_examples = [
            example
            for example in dataset_examples
            if split.contains( example.problem_path )
        ]

        dataset_skipped = [
            item
            for item in dataset_skipped
            if split.contains(item.get("problem", ""))
        ]

        log_message(
            f"evaluating {len(dataset_examples)} of {total} dataset examples ({split.describe()}) without proof search"
        )

        if not dataset_examples:
            raise RuntimeError(f"no examples in {split.describe()} of {dataset}")

        sat_policy = label_policy(collection_config.sat_policy, metadata)
        problem_list = [example.problem_path for example in dataset_examples]

    tracker = WandbTracker.start(
        replace(
            wandb_config,
            name = wandb_config.name or f"{type(predictor.model).__name__}-evaluate",
            name_prefix = None,
        ),
        job_type = "evaluate",
        run_config = {
            "checkpoint": str(checkpoint),
            "model_config": predictor.model.config.to_dict(),
            "device": str(predictor.device),

            "sat_policy": sat_policy.wire_value,
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
                tptp_root = tptp_root,
                config = collection_config,
                progress = None if tracker is None else tracker.log_collection,
                resume = resume,
            )
        else:
            examples, skipped = dataset_examples, dataset_skipped

        report_dataset(
            dataset_summary(examples, skipped, batch_size = collection_config.batch_size)
        )

        started = time.monotonic()
        _, probabilities, raw_metrics = example_outputs(
            predictor.model,
            examples,
            batch_size = collection_config.batch_size,
            adaptive = multiprocess,
            resume = resume,
        )

        metrics: dict[str, Any] = dict(raw_metrics)
        report_problem_predictions(examples, probabilities, skipped, split)
        log_message(
            f"evaluated {len(examples)} problems in {time.monotonic() - started:.1f}s: {metric_text(metrics)}"
        )

        metrics.update(
            {
                "evaluation_kind": "labelled SAT-core-membership evaluation",
                "model_config": predictor.model.config.to_dict(),
                "label_semantics": "native CaDiCaL failed-assumption SAT-core membership",
                "sat_policy": sat_policy.wire_value,

                "dataset": None if dataset is None else str(dataset),
                "split": split.to_dict(),
                "training_split": predictor.training_config.get("split"),

                "problems_proved": len(examples),
                "problems_skipped": len(skipped),
                "skipped": skipped,
            }
        )

        if tracker is not None:
            tracker.log_results(examples, probabilities, metrics, split = split)

        return metrics
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code = 1)
            tracker = None

        raise
    finally:
        if tracker is not None:
            tracker.finish()

def report_problem_predictions(examples, probabilities, skipped, split):
    offset = 0
    for example in examples:
        size = len(example.labels)
        scores = probabilities[offset : offset + size]
        metrics = prediction_metrics(
            [int(label) for label in example.labels],
            scores,
            problem_sizes = [size],
        )

        emit(
            "problem",
            problem = example.problem_path,
            part = split.part(example.problem_path),
            outcome = "evaluated",
            **metrics,
        )

        offset += size

    for item in skipped:
        emit("problem", **item, part = split.part(item["problem"]))

def warn_on_training_overlap(training_config: Mapping[str, Any], split: ProblemSplit):
    trained = training_config.get("split")
    if not isinstance(trained, Mapping):
        log_message(
            "warning: the checkpoint does not record its training split, so held-out evaluation cannot be checked"
        )

        return

    trained_split = ProblemSplit(
        int(trained.get("split", 1)),
        tuple(int(p) for p in trained.get("parts", (0,))),
        str(trained.get("by", SplitKey.Family)),
    )

    if not trained_split.is_everything and trained.get("scheme") != SPLIT_SCHEME:
        log_message(
            f"warning: the checkpoint was trained on {trained_split.describe()} with an older split hash; its parts are not the parts this version selects, so problems may overlap"
        )
    elif trained_split.is_everything:
        log_message(
            "warning: the checkpoint was trained on every problem of its dataset; metrics on problems from that dataset are not held-out"
        )
    elif (trained_split.split, trained_split.by) != (split.split, split.by):
        log_message(
            f"warning: the checkpoint was trained on {trained_split.describe()}, a different split from {split.describe()}; problems may overlap"
        )
    elif set(trained_split.parts) & set(split.parts):
        log_message(
            f"warning: evaluating {split.describe()} overlaps the training {trained_split.describe()}; metrics are not held-out"
        )

def evaluate_examples(
    model: AxiomPredictionNetwork,
    examples: list[AxiomTrainingExample],
    *,
    device: torch.device,
) -> dict[str, float | int | None]:
    _ = device
    return example_outputs(model, examples, batch_size = 43)[2]

def predict_examples(model, examples, *, batch_size, adaptive):
    from .multiprocess import AdaptiveBatches, adaptive_predictions, predict_graphs

    if adaptive:
        graphs = [example.graph for example in examples]
        yield from adaptive_predictions(model, graphs, AdaptiveBatches(len(examples)))
        return

    batches = math.ceil(len(examples) / batch_size)
    for chunk in track_progress(chunks(examples, batch_size), "evaluating batches", batches):
        yield from predict_graphs(model, [example.graph for example in chunk])

def example_outputs(
    model: AxiomPredictionNetwork,
    examples: list[AxiomTrainingExample],
    *,
    batch_size: int,
    adaptive: bool = False,
    resume = None,
) -> tuple[list[int], list[float], dict[str, float | int | None]]:
    model.eval()
    predict = partial(predict_examples, model, batch_size = batch_size, adaptive = adaptive)
    outputs = predict(examples) if resume is None else resume_predictions(examples, predict, resume)
    probabilities = [value for values in outputs for value in values]
    labels = [int(label) for example in examples for label in example.labels]

    log_message(f"computing prediction metrics for {len(labels)} axiom scores")
    metrics = prediction_metrics(
        labels,
        probabilities,
        problem_sizes = [len(example.labels) for example in examples],
    )

    return labels, probabilities, metrics

def log_message(message: str):
    log(f"axiom-predictor: {message}")

def emit(event: str, **fields: object):
    write_record({"event": event, **fields})

def dataset_summary(
    examples: list[AxiomTrainingExample],
    skipped: list[dict[str, str]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    sizes = [len(example.labels) for example in examples]
    cores = [int(sum(example.labels)) for example in examples]

    nodes = [
        sum(len(rows) for rows in example.graph.graph.nodes.values())
        for example in examples
    ]

    edges = [
        sum(len(rows) for rows in example.graph.graph.edges.values())
        for example in examples
    ]

    labels = sum(sizes)
    largest = sorted(zip(examples, sizes, nodes), key = lambda item: -item[1])[:5]

    return {
        "problems": len(examples),
        "duplicate_problems": len(examples) - len(
            {example.problem_path for example in examples}
        ),
        "problems_skipped": len(skipped),
        "skipped_outcomes": dict(
            Counter(
                outcome_kind(item.get("outcome", "")) for item in skipped
            ).most_common()
        ),
        "batches_per_epoch": math.ceil(len(examples) / batch_size),

        "axiom_clauses": labels,
        "core_clauses": sum(cores),
        "positive_rate": sum(cores) / labels if labels else None,

        "axioms_per_problem": spread(sizes),
        "core_per_problem": spread(cores),
        "core_fraction_per_problem": spread([c / n for c, n in zip(cores, sizes) if n]),

        "problems_without_axioms": sum(n == 0 for n in sizes),
        "problems_with_empty_core": sum(c == 0 for c in cores),
        "problems_with_full_core": sum(c == n and n > 0 for c, n in zip(cores, sizes)),

        "graph_nodes_per_problem": spread(nodes),
        "graph_edges_per_problem": spread(edges),
        "largest_problems": [
            {"problem": example.problem_path, "axiom_clauses": n, "graph_nodes": g}
            for example, n, g in largest
        ],
    }

def report_dataset(summary: dict[str, Any]):
    rate = summary["positive_rate"]
    log_message(
        f"dataset: {summary['problems']} problems ({summary['problems_skipped']} skipped, {summary['duplicate_problems']} duplicates), {summary['axiom_clauses']} axiom clauses, {summary['core_clauses']} in SAT cores ({'n/a' if rate is None else f'{100 * rate:.2f}%'}), {summary['batches_per_epoch']} batches per epoch"
    )

    if summary["skipped_outcomes"]:
        log_message(
            "skipped outcomes: "
            + ", ".join(
                f"{kind} x{count}" for kind, count in summary["skipped_outcomes"].items()
            )
        )

    log_message(f"axiom clauses per problem: {spread_text(summary['axioms_per_problem'])}")
    log_message(
        f"SAT-core clauses per problem: {spread_text(summary['core_per_problem'])}; core fraction {spread_text(summary['core_fraction_per_problem'])}"
    )

    log_message(
        f"{summary['problems_with_empty_core']} problems with an empty axiom core, {summary['problems_with_full_core']} with every axiom in the core, {summary['problems_without_axioms']} without axioms"
    )

    log_message(
        f"graph nodes per problem: {spread_text(summary['graph_nodes_per_problem'])}; edges per problem: {spread_text(summary['graph_edges_per_problem'])}"
    )

    log_message(
        "largest problems: "
        + ", ".join(
            f"{item['problem']} ({item['axiom_clauses']} axioms, {item['graph_nodes']} nodes)"
            for item in summary["largest_problems"]
        )
    )

    emit("dataset", **summary)

def report_progress(
    epoch: int,
    epochs: int,
    index: int,
    batches: int,

    *,
    loss: float,
    seconds: float,
    labels: int,
):
    log_message(
        f"epoch {epoch}/{epochs}: batch {index}/{batches} done after {seconds:.1f}s, batch loss {loss:.6f} over {labels} axiom clauses"
    )

    emit(
        "progress",
        epoch = epoch,
        epochs = epochs,
        batch = index,
        batches = batches,

        loss = loss,
        seconds = seconds,
        labels = labels,
    )

def report_epoch(
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
        else f"; train-set {metric_text(metrics)} (evaluated in {eval_seconds:.2f}s)"
    )

    log_message(
        f"epoch {epoch}/{epochs}: loss {loss:.6f}, mean grad norm {'n/a' if norm is None else f'{norm:.4f}'}, {len(norms)} batches in {seconds:.2f}s{suffix}{warning}"
    )

    emit(
        "epoch",
        epoch = epoch,
        epochs = epochs,

        loss = loss,
        seconds = seconds,
        eval_seconds = eval_seconds,

        batches = len(norms),
        grad_norm_mean = norm,
        grad_norm_max = max(norms, default = None),

        metrics = metrics,
    )

def metric_text(metrics: Mapping[str, object]) -> str:
    names = (
        "bce",
        "roc_auc",
        "average_precision",
        "macro_average_precision",
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

def spread(values: list[float] | list[int]) -> dict[str, float] | None:
    if not values:
        return None

    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }

def spread_text(spread: object) -> str:
    if not isinstance(spread, Mapping):
        return "n/a"

    return ", ".join(f"{name} {value:.3g}" for name, value in spread.items())

def outcome_kind(outcome: str) -> str:
    head, _, rest = outcome.partition(": ")
    code = re.match(r"\s*(\[[A-Z0-9_]+\])", rest)
    return head if code is None else f"{head} {code.group(1)}"

def chunks(examples: list[AxiomTrainingExample], size: int):
    if size < 1:
        raise ValueError("batch size must be at least 1")

    for start in range(0, len(examples), size):
        yield examples[start : start + size]

def write_json(path: Path, value: object):
    write_durably(
        path,
        (json.dumps(value, indent = 2, sort_keys = True) + "\n").encode("utf-8"),
    )

__all__ = [
    "AxiomTrainingConfig",
    "collect_examples",
    "evaluate_axiom_predictor",
    "evaluate_examples",
    "train_axiom_predictor",
]
