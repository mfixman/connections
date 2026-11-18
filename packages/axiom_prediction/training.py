# `train`: fits a network to the SAT-core labels of a joined dataset, saving a resumable checkpoint every epoch.

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn

from .background_validation import BackgroundValidation
from .choices import GraphInputKind, LossWeighting, ProverPolicy, plain_values
from .data import AxiomTrainingExample
from .dataset import label_policy, load_axiom_dataset
from .graph import collate_axiom_graphs
from .limits import default_step_limit, default_timeout_s
from .logs import log
from .metrics import prediction_metrics
from .model import AxiomModelConfig, AxiomPredictionNetwork, resolve_checkpoint, resolve_device, save_checkpoint, write_durably
from .models import load_model_class
from .parallel import determine_worker_count
from .resume import check_training_target, dataset_fingerprint, next_training_run, restore_training, snapshot_training
from .split import ProblemSplit
from .training_log import log_dataset, log_epoch, metric_text
from .validation_run import subset_sizes, validation_run_configs, validation_run_problems
from .wandb_tracking import WandbConfig, WandbTracker

# Saved in every checkpoint; a resumed run must repeat it, so new fields would break old checkpoints.
@dataclass(slots = True, init = False)
class AxiomTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float

    hidden_dim: int
    message_rounds: int
    num_hidden_layers: int
    activation: str
    graph_input: str
    model: str | None

    seed: int
    device: str

    # Limits of the validation proof searches.
    step_limit: int
    timeout_s: float
    sat_policy: ProverPolicy | None
    num_workers: int | None

    def __init__(
        self,
        epochs = 200,
        batch_size = 43,
        learning_rate = 1e-3,
        weight_decay = 0.0,
        hidden_dim = 64,
        message_rounds = 3,
        num_hidden_layers = 2,
        activation = "relu",
        graph_input = GraphInputKind.Full,
        model = None,
        seed = 0,
        device = "cuda",
        step_limit = default_step_limit,
        timeout_s = default_timeout_s,
        sat_policy = None,
        num_workers = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.hidden_dim = hidden_dim
        self.message_rounds = message_rounds
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.graph_input = graph_input
        self.model = model

        self.seed = seed
        self.device = device
        self.step_limit = step_limit
        self.timeout_s = timeout_s
        self.sat_policy = None if sat_policy is None else ProverPolicy(sat_policy)
        self.num_workers = num_workers

    def model_class(self):
        return AxiomPredictionNetwork if self.model is None else load_model_class(self.model)

    # A named network brings its own sizes; the fields above describe the unnamed base network.
    def network_config(self) -> AxiomModelConfig:
        if self.model is not None:
            return replace(self.model_class().default_config)

        return AxiomModelConfig(
            hidden_dim = self.hidden_dim,
            message_rounds = self.message_rounds,
            num_hidden_layers = self.num_hidden_layers,
            activation = self.activation,
            graph_input = self.graph_input,
        )

# Each problem counts once, however many clauses it has; weights share a problem between its cores.
def problem_loss(logits, labels, counts, weights):
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction = "none")
    means = torch.stack([part.mean() for part in losses.split(counts)])
    return (means * weights).sum() / weights.sum()

def split_examples(examples, failures, split):
    kept = [example for example in examples if split.contains(example.problem_path)]
    return kept, [item for item in failures if split.contains(item.get("problem", ""))]

def saved_config(config, policy, split, loss_weighting, min_axioms, metadata):
    payload = plain_values(asdict(config))
    payload["timeout_seconds"] = payload.pop("timeout_s")
    payload.update(config.network_config().to_dict())
    payload["sat_policy"] = policy.wire_value()
    payload["split"] = split.to_dict()
    # Resume compares every key, so defaults stay out for earlier checkpoints.
    if loss_weighting != LossWeighting.Clause:
        payload["loss_weighting"] = loss_weighting.wire_value()

    if min_axioms:
        payload["min_axioms"] = min_axioms

    if policy == ProverPolicy.SatCoP:
        from .satcop import label_semantics

        payload["collection"] = metadata.get("collection") or {
            "sat_policy": "satcop",
            "prover_backend": "rust-satcop",
            "label_semantics": label_semantics,
        }

    return payload

def train_axiom_predictor(
    *,
    dataset,
    output_dir,
    tptp_root = None,
    config = None,
    wandb_config = WandbConfig(),
    run_properties = None,
    split = None,
    evaluation_split = None,
    evaluate_every = 1,
    validation_run_ratio = 0,
    validation_run_provers = (),
    resume = False,
    max_example_mb = None,
    loss_weighting = LossWeighting.Clause,
    min_axioms = 0,
    validation_all_held_out = False,
) -> dict[str, Any]:
    train_metrics_every = 10

    config = AxiomTrainingConfig() if config is None else config
    split = ProblemSplit() if split is None else split
    output = Path(output_dir)
    check_training_target(output, resume)
    if evaluation_split is not None and (
        split.split != evaluation_split.split
        or split.scheme != evaluation_split.scheme
        or set(split.parts) & set(evaluation_split.parts)
    ):
        raise ValueError("evaluation parts must use the training split and not overlap --parts")

    run_configs = []
    if evaluation_split is not None and validation_run_ratio:
        run_configs = validation_run_configs(validation_run_provers, config, output / "model.ts")

    log(f"loading dataset from {dataset}")
    everything, failures, metadata = load_axiom_dataset(dataset, max_example_mb)
    held_out, held_out_skipped = [], []
    if evaluation_split is not None:
        held_out, held_out_skipped = split_examples(everything, failures, evaluation_split)
        if not held_out:
            raise ValueError(f"no evaluation examples in {evaluation_split.describe()}")

    examples, skipped = split_examples(everything, failures, split)
    log(f"kept {len(examples)} of {len(everything)} examples in {split.describe()}; collection settings {metadata.get('collection')}")
    if not examples:
        raise RuntimeError(f"no training examples in {split.describe()} of {dataset}")

    policy = label_policy(config.sat_policy, metadata)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    output.mkdir(parents = True, exist_ok = True)
    payload = saved_config(config, policy, split, loss_weighting, min_axioms, metadata)

    run_count = next_training_run(output, resume)
    run_name = config.model or AxiomPredictionNetwork.__name__
    tracker = WandbTracker.start(
        replace(wandb_config, name = wandb_config.name or (run_name if run_count == 1 else f"{run_name}-{run_count}")),
        job_type = "train",
        run_config = {
            **payload,
            "run_count": run_count,
            "resolved_device": str(device),
            "problems": [example.problem_path for example in examples],
            "problems_requested": len(examples),
            "dataset": str(dataset),
            "evaluation_split": None if evaluation_split is None else evaluation_split.to_dict(),
            "evaluate_every": evaluate_every,
            "validation_run_ratio": validation_run_ratio,
            "validation_run_provers": [c.policy.wire_value() for c in run_configs],
            "cli_arguments": dict(run_properties or {}),
        },
        output_dir = output,
        resume = resume,
    )

    background = None
    try:
        if loss_weighting == LossWeighting.Problem or min_axioms:
            total = len(examples)
            examples = [example for example in examples if len(example.labels) > min_axioms]
            log(f"kept {len(examples)} of {total} examples with more than {min_axioms} axioms")

        training_workers = determine_worker_count(None, config.num_workers)
        if run_configs:
            background = start_validation(
                held_out,
                evaluation_split,
                validation_run_ratio,
                validation_all_held_out,
                run_configs,
                tracker,
                training_workers,
                tptp_root,
            )

            training_workers = 1

        if tracker is not None:
            counts = {"collection/problems_proved": len(examples), "collection/problems_skipped": len(skipped)}
            tracker.run.summary.update(counts)

        log_dataset(examples, skipped, config.batch_size)
        torch.set_num_threads(training_workers)
        model = config.model_class()(config.network_config()).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        log(f"model: {payload}; {parameters} parameters on {device} with {training_workers} Torch CPU threads")

        optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
        rnd = random.Random(config.seed)
        fingerprint = dataset_fingerprint(examples)
        first_epoch = 1
        if resume:
            log(f"restoring optimizer and RNG state from {resolve_checkpoint(output)}")
            first_epoch = restore_training(resolve_checkpoint(output), model, optimizer, rnd, payload, fingerprint)

        write_json(output / "training_config.json", payload)
        core_counts = Counter(example.problem_path for example in examples)

        def loss_of(logits, labels, chunk, counts):
            if loss_weighting == LossWeighting.Clause:
                return nn.functional.binary_cross_entropy_with_logits(logits, labels)

            weights = torch.tensor([1 / core_counts[example.problem_path] for example in chunk], device = device)
            return problem_loss(logits, labels, counts, weights)

        probabilities, train_metrics = [], None
        evaluation_probabilities, evaluation_metrics, last_evaluation = [], None, None
        for epoch in range(first_epoch, config.epochs + 1):
            started = time.monotonic()
            log(f"epoch {epoch}/{config.epochs}: starting")
            order = list(examples)
            rnd.shuffle(order)
            loss, norms = train_epoch(model, optimizer, order, config.batch_size, loss_of, device, epoch, tracker)
            train_s = time.monotonic() - started

            evaluation = None
            last_epoch = epoch == config.epochs
            if held_out and (last_evaluation is None or last_epoch or epoch - last_evaluation >= evaluate_every):
                _, evaluation_probabilities, evaluation_metrics = example_outputs(model, held_out, batch_size = config.batch_size)
                evaluation, last_evaluation = dict(evaluation_metrics), epoch
                log(f"epoch {epoch}: held-out {metric_text(evaluation_metrics)} ({len(held_out)} problems)")

            # Scoring the whole training set costs about an epoch, so it happens only now and then.
            train_metrics = None
            if epoch == 1 or last_epoch or epoch % train_metrics_every == 0:
                _, probabilities, train_metrics = example_outputs(model, examples, batch_size = config.batch_size)

            log_epoch(epoch, config.epochs, loss, train_s, norms, train_metrics)
            epoch_path = output / f"epoch-{epoch:04d}.ts"
            save_checkpoint(
                epoch_path,
                model,
                training_config = payload,
                epoch = epoch,
                training_state = snapshot_training(optimizer, rnd, fingerprint),
            )

            write_durably(output / "model.ts", epoch_path.read_bytes())
            log(f"saved epoch {epoch} checkpoint to {epoch_path}")
            if background is not None:
                background.submit(epoch, epoch_path)

            if tracker is not None:
                tracker.log_epoch(
                    epoch,
                    loss = loss,
                    duration_s = train_s,
                    grad_norm = sum(norms) / len(norms) if norms else None,
                    metrics = train_metrics,
                    evaluation = evaluation,
                )

        if train_metrics is None:
            _, probabilities, train_metrics = example_outputs(model, examples, batch_size = config.batch_size)

        metrics = {
            **train_metrics,
            "evaluation_kind": "training-set overfit (not held-out generalization)",
            "label_semantics": payload.get("collection", {}).get(
                "label_semantics", "native CaDiCaL failed-assumption SAT-core membership"
            ),
            "sat_policy": policy.wire_value(),
            "dataset": str(dataset),
            "problems_proved": len(examples),
            "problems_skipped": len(skipped),
            "skipped": skipped,
        }

        write_json(output / "metrics.json", metrics)
        if evaluation_metrics is not None:
            evaluation_metrics = {
                **evaluation_metrics,
                "evaluation_kind": "held-out SAT-core-membership evaluation",
                "split": evaluation_split.to_dict(),
                "epoch": last_evaluation,
                "problems_proved": len(held_out),
                "skipped": held_out_skipped,
            }

            write_json(output / "evaluation_metrics.json", evaluation_metrics)

        log(f"saved checkpoint, config and metrics to {output}")
        if background is not None:
            log("waiting for the final validation run")
            background.finish()
            background = None

        if tracker is not None:
            tracker.log_results(examples, probabilities, metrics, split = split)
            if evaluation_metrics is not None:
                tracker.log_evaluation_results(held_out, evaluation_probabilities, evaluation_metrics, split = evaluation_split)

            tracker.log_model(output / "model.ts")

        return metrics
    except BaseException:
        if background is not None:
            background.finish(abort = True)

        if tracker is not None:
            tracker.finish(exit_code = 1)
            tracker = None

        raise
    finally:
        if tracker is not None:
            tracker.finish()

# Proof search on held-out problems runs beside training, on every CPU but one.
def start_validation(held_out, split, ratio, all_held_out, run_configs, tracker, cpus, tptp_root):
    problems = validation_run_problems(held_out, split, ratio, tptp_root, all_held_out = all_held_out)
    log(f"validation runs: {len(problems)} held-out problems per prover for each epoch")
    if tracker is not None:
        constants = subset_sizes(problems, held_out, run_configs)
        tracker.run.summary.update({f"evaluation/{k}": v for k, v in constants.items()})

    return BackgroundValidation(problems, run_configs, tracker, count = cpus, tptp_root = tptp_root)

# One pass over the shuffled examples; returns the mean loss per axiom clause and each batch's gradient norm.
def train_epoch(model, optimizer, order, batch_size, loss_of, device, epoch, tracker):
    progress_interval_s = 60.0

    model.train()
    batches = math.ceil(len(order) / batch_size)
    total_loss, total_labels, norms = 0.0, 0, []
    last_progress = time.monotonic()
    for index, chunk in enumerate(chunks(order, batch_size), start = 1):
        batch = collate_axiom_graphs([(example.graph, example.labels) for example in chunk])
        if len(batch.labels) == 0:
            continue

        labels = batch.labels.to(device)
        optimizer.zero_grad()
        loss = loss_of(model(batch), labels, chunk, batch.axiom_counts)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss; checkpoint unchanged")

        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite = True)))
        optimizer.step()

        batch_loss = float(loss.detach().cpu().item())
        total_loss += batch_loss * len(labels)
        total_labels += len(labels)
        if index < batches and time.monotonic() - last_progress >= progress_interval_s:
            last_progress = time.monotonic()
            log(f"epoch {epoch}: batch {index}/{batches}, batch loss {batch_loss:.6f} over {len(labels)} axiom clauses")
            if tracker is not None:
                tracker.log_progress(epoch - 1 + index / batches, loss = batch_loss, batches_done = index, batches_total = batches)

    return (total_loss / total_labels if total_labels else None), norms

# The probabilities of each graph's axiom clauses, in one batched forward pass.
def predict_graphs(model, graphs):
    with torch.inference_mode():
        logits = model(collate_axiom_graphs([(graph, None) for graph in graphs]))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced non-finite logits")

        values = torch.sigmoid(logits).cpu()

    return [part.tolist() for part in values.split([len(graph.axiom_clause_ids) for graph in graphs])]

# Returns the labels, the probabilities and their metrics over all axiom clauses of the examples.
def example_outputs(model, examples, *, batch_size):
    model.eval()
    probabilities = [
        value
        for chunk in chunks(examples, batch_size)
        for values in predict_graphs(model, [example.graph for example in chunk])
        for value in values
    ]

    labels = [int(label) for example in examples for label in example.labels]
    sizes = [len(example.labels) for example in examples]
    return labels, probabilities, prediction_metrics(labels, probabilities, problem_sizes = sizes)

def chunks(examples: list[AxiomTrainingExample], size: int):
    for start in range(0, len(examples), size):
        yield examples[start : start + size]

def write_json(path: Path, value: Mapping[str, Any]):
    write_durably(path, (json.dumps(value, indent = 2, sort_keys = True) + "\n").encode("utf-8"))
