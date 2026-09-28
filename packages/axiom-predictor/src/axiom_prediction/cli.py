from __future__ import annotations

from .choices import GuidanceMode, ProverPolicy

from typing import Any

import argparse
import json
from pathlib import Path
import re

from .model import device_description
from .models import available_models
from connections.parsing.tptp import TPTPParseError

from .dataset import NoParseableProblemsError, collect_axiom_dataset, collect_axiom_dataset_shard
from .logs import log
from .tptp import (
    DEFAULT_STEP_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,

    NoProblemFilesError,
    expand_problem_inputs,
    load_tptp_problem,

    problem_input_directory,
    tptp_problems,
)

from .run import RUN_POLICIES
from .split import ProblemSplit
from .wandb_tracking import WandbConfig

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "axiom-predictor",
        epilog = f"Training networks (--network): {', '.join(available_models())}",
    )

    subparsers = parser.add_subparsers(dest = "command", required = True)

    train = subparsers.add_parser("train", help = "train from proofs of TPTP problems")
    train.add_argument("problems", metavar = "PROBLEM", nargs = "*")
    train.add_argument(
        "--data-dir",
        type = Path,
        required = True,
        help = "run directory containing dataset/ and receiving model/ (or models/NAME with --model-name)",
    )

    add_model_name_argument(train)
    add_dataset_argument(train)
    train.add_argument(
        "--network",
        default = "DefaultFull",
        help = f"network module from models/: {', '.join(available_models())}",
    )

    train.add_argument(
        "--epochs",
        type = positive_int,
        default = 200,
        help = "passes through the training examples",
    )

    train.add_argument("--tptp", type = Path)
    add_split_arguments(train)
    train.add_argument("--seed", type = int, default = 0)
    train.add_argument(
        "--batch-size",
        type = positive_int,
        default = 43,
        help = "problems per optimizer update, not axiom clauses",
    )

    train.add_argument(
        "--log-every",
        type = nonnegative_int,
        default = 10,
        metavar = "EPOCHS",
        help = "print train-set metrics every EPOCHS epochs (0 disables; loss is printed every epoch)",
    )

    add_device_argument(train)
    train.add_argument(
        "--step-limit",
        type = nonnegative_int,
        default = DEFAULT_STEP_LIMIT,
    )

    train.add_argument(
        "--timeout-seconds",
        type = float,
        default = DEFAULT_TIMEOUT_SECONDS,
    )

    train.add_argument(
        "--policy",
        type = ProverPolicy,
        choices = tuple(ProverPolicy),
        help = "label collection policy; defaults to dataset provenance or satresetcop",
    )

    add_worker_argument(train)
    add_wandb_arguments(train)

    collect = subparsers.add_parser(
        "collect",
        help = "collect a resumable SAT-core training dataset",
    )

    collect.add_argument("problems", metavar = "PROBLEM", nargs = "*")
    collect.add_argument(
        "--data-dir",
        type = Path,
        help = "write under DATA_DIR/dataset unless --dataset is supplied",
    )

    add_dataset_argument(collect)
    collect.add_argument("--tptp", type = Path)
    add_split_arguments(collect)
    collect.add_argument(
        "--step-limit",
        type = nonnegative_int,
        default = DEFAULT_STEP_LIMIT,
    )

    collect.add_argument(
        "--timeout-seconds",
        type = float,
        default = DEFAULT_TIMEOUT_SECONDS,
    )

    collect.add_argument(
        "--policy",
        type = ProverPolicy,
        choices = tuple(ProverPolicy),
        default = ProverPolicy.SatResetCoP,
    )

    add_worker_argument(collect)

    evaluate = subparsers.add_parser(
        "evaluate",
        help = "score SAT-core predictions against a collected dataset or fresh proofs",
    )

    evaluate.add_argument(
        "checkpoint",
        metavar = "CHECKPOINT",
        type = Path,
        nargs = "?",
        help = "model.pt or its directory; defaults to the --data-dir model selected by --model-name",
    )

    evaluate.add_argument("problems", metavar = "PROBLEM", nargs = "*")
    evaluate.add_argument(
        "--data-dir",
        type = Path,
        help = "score examples in DATA_DIR/dataset without proof search instead of proving PROBLEM inputs",
    )

    add_model_name_argument(evaluate)
    add_dataset_argument(evaluate)
    evaluate.add_argument("--tptp", type = Path)
    add_split_arguments(evaluate)
    add_device_argument(evaluate)
    evaluate.add_argument(
        "--batch-size",
        type = positive_int,
        default = 43,
        help = "problems per prediction batch",
    )

    evaluate.add_argument(
        "--step-limit",
        type = nonnegative_int,
        default = DEFAULT_STEP_LIMIT,
    )

    evaluate.add_argument(
        "--timeout-seconds",
        type = float,
        default = DEFAULT_TIMEOUT_SECONDS,
    )

    evaluate.add_argument(
        "--policy",
        type = ProverPolicy,
        choices = tuple(ProverPolicy),
        help = "label collection policy; defaults to dataset provenance or satresetcop",
    )

    add_worker_argument(evaluate)
    add_wandb_arguments(evaluate)

    run = subparsers.add_parser(
        "run",
        help = "run a SAT policy on problems, optionally ordering start and extension clauses by the axiom predictor",
    )

    run.add_argument("problems", metavar = "PROBLEM", nargs = "*")
    run.add_argument(
        "--model",
        type = Path,
        help = "trained checkpoint (model.pt or its directory); without a model, run the unguided baseline",
    )

    run.add_argument(
        "--data-dir",
        type = Path,
        help = "take the model from DATA_DIR/model, or DATA_DIR/models/NAME with --model-name",
    )

    add_model_name_argument(run)
    run.add_argument(
        "--policy",
        type = ProverPolicy,
        choices = RUN_POLICIES,
        default = ProverPolicy.SatResetCoP,
    )

    run.add_argument("--seed", type = int, default = 0)
    run.add_argument("--tptp", type = Path)

    add_device_argument(run)
    run.add_argument(
        "--step-limit",
        type = nonnegative_int,
        default = DEFAULT_STEP_LIMIT,
    )

    run.add_argument("--timeout-seconds", type = float, default = DEFAULT_TIMEOUT_SECONDS)

    add_split_arguments(run)
    add_worker_argument(run)
    add_wandb_arguments(run)

    predict = subparsers.add_parser(
        "predict",
        help = "rank every clausified axiom clause",
    )

    predict.add_argument(
        "checkpoint",
        metavar = "CHECKPOINT",
        type = Path,
        nargs = "?",
        help = "model.pt or its directory; omit it when using --data-dir",
    )

    predict.add_argument("problems", metavar = "PROBLEM", nargs = "*")
    predict.add_argument(
        "--data-dir",
        type = Path,
        help = "take the model from this run directory instead of CHECKPOINT",
    )

    add_model_name_argument(predict)
    predict.add_argument("--tptp", type = Path)
    add_split_arguments(predict)
    add_device_argument(predict)
    common_help = {
        "problems": "problem files, flat directories or TPTP categories; train with no inputs uses DATA_DIR/dataset",
        "tptp": "TPTP root; defaults to $TPTP, then conventional TPTP/corpora directories near the checkout or in your home",
        "seed": "random seed for model initialization and training shuffle, or proof-search choices",
        "step_limit": "maximum prover steps per problem (fresh proof search only)",
        "timeout_seconds": "wall-clock budget in seconds per problem (fresh proof search only)",
        "policy": "SAT policy used for proof search and label collection",
    }

    for command in subparsers.choices.values():
        command.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        for action in command._actions:
            if action.help is None and action.dest in common_help:
                action.help = common_help[action.dest]

    return parser

def add_device_argument(parser):
    parser.add_argument(
        "--device",
        default = "cuda",
        help = "cuda/cuda:N requires a GPU; auto selects CUDA if available and CPU otherwise; cpu forces CPU",
    )

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            dataset = selected_dataset(args)
            if dataset is None:
                raise ValueError("collect needs --dataset or --data-dir")

            problems = selected_problems(args)
            input_directory = (
                problem_input_directory(args.problems[0], tptp_root = args.tptp)
                if len(args.problems) == 1
                else None
            )

            collection_args: dict[str, Any] = {
                "output_dir": dataset,
                "tptp_root": args.tptp,
                "step_limit": args.step_limit,
                "timeout_seconds": args.timeout_seconds,
                "sat_policy": args.policy,
                "num_workers": args.num_workers,
            }

            if input_directory is None and args.dataset is None:
                summary = collect_axiom_dataset(problems, **collection_args)
            else:
                shard, summary = collect_axiom_dataset_shard(
                    problems,
                    shard_name = (
                        "problems" if input_directory is None else input_directory.name
                    ),
                    **collection_args,
                )

                summary = {**summary, "dataset_shard": str(shard)}

            log(json.dumps(summary, sort_keys = True))
            return 0

        if args.command == "run":
            return run_command(args)

        print_device(args.device)
        if args.command == "train":
            from .training import AxiomTrainingConfig, train_axiom_predictor

            problems, dataset = training_request(args)
            args.data_dir.mkdir(parents = True, exist_ok = True)
            metrics = train_axiom_predictor(
                problems,
                dataset = dataset,
                output_dir = model_directory(args.data_dir, args.model_name),
                tptp_root = args.tptp,
                config = AxiomTrainingConfig(
                    model = args.network,
                    epochs = args.epochs,
                    device = args.device,

                    seed = args.seed,
                    batch_size = args.batch_size,
                    log_every = args.log_every,

                    step_limit = args.step_limit,
                    timeout_seconds = args.timeout_seconds,
                    sat_policy = args.policy,
                    num_workers = args.num_workers,
                ),
                wandb_config = wandb_config(args),
                run_properties = cli_properties(args),
                split = selected_split(args),
            )

            log(json.dumps(metrics, sort_keys = True))
            return 0

        if args.command == "evaluate":
            from .training import AxiomTrainingConfig, evaluate_axiom_predictor

            dataset = selected_dataset(args)
            if dataset is not None and args.problems:
                raise ValueError("evaluate takes PROBLEM inputs or a dataset, not both")

            checkpoint = selected_checkpoint(args)
            problems = () if dataset is not None else selected_problems(args)
            metrics = evaluate_axiom_predictor(
                checkpoint,
                list(problems),
                dataset = dataset,
                split = selected_split(args),

                tptp_root = args.tptp,
                device = args.device,
                config = AxiomTrainingConfig(
                    device = args.device,
                    batch_size = args.batch_size,
                    step_limit = args.step_limit,
                    timeout_seconds = args.timeout_seconds,
                    sat_policy = args.policy,
                    num_workers = args.num_workers,
                ),
                wandb_config = wandb_config(args),
                run_properties = cli_properties(args),
            )

            print(json.dumps(metrics, sort_keys = True))
            return 0

        from .model import AxiomPredictor

        if args.data_dir is not None and args.checkpoint is not None:
            args.problems = [str(args.checkpoint), *args.problems]
            args.checkpoint = None

        predictor = AxiomPredictor.load(selected_checkpoint(args), device = args.device)
        problems = selected_problems(args)
        parseable = 0
        for problem in problems:
            try:
                loaded = load_tptp_problem(problem, tptp_root = args.tptp)
            except TPTPParseError as error:
                log(f"skipping {problem}: {error}")
                continue

            parseable += 1
            predictions = predictor.predict(
                loaded.matrix,
                axiom_clause_ids = loaded.axiom_clause_ids,
                conjecture_clause_ids = loaded.conjecture_clause_ids,
            )

            for prediction in predictions:
                print(
                    json.dumps(
                        {
                            "problem_path": loaded.requested_path,
                            "clause_index": prediction.clause_index,
                            "clause_text": prediction.clause_text,
                            "probability": prediction.probability,
                            "rank": prediction.rank,
                        },
                        sort_keys = True,
                    )
                )

            log(f"ranked {len(predictions)} axiom clauses for {loaded.requested_path}")

        if not parseable:
            raise NoParseableProblemsError(
                f"none of the {len(problems)} .p files could be parsed as FOF or CNF"
            )

        return 0
    except (NoProblemFilesError, NoParseableProblemsError) as error:
        log(f"axiom-predictor: error: {error}")
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        log(f"axiom-predictor: error: {error}")
        return 2

def print_device(requested: str):

    description = device_description(requested)
    suffix = " (automatically selected)" if requested in ("gpu", "auto") else ""
    log(f"axiom-predictor: COMPUTE DEVICE: {description}{suffix}")
    if requested in ("gpu", "auto") and description == "CPU":
        log("warning: CUDA is unavailable; using CPU")

def model_directory(data_dir: Path, model_name: str | None) -> Path:
    if model_name is None:
        return data_dir / "model"

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name):
        raise ValueError(
            f"--model-name must be letters, digits, '.', '_' or '-', starting with a letter or digit: {model_name!r}"
        )

    return data_dir / "models" / model_name

def selected_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        if args.model_name is not None:
            raise ValueError("give either CHECKPOINT or --model-name, not both")

        return args.checkpoint

    if args.data_dir is None:
        raise ValueError(f"{args.command} needs a CHECKPOINT or --data-dir")

    return model_directory(args.data_dir, args.model_name)

def add_model_name_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        metavar = "NAME",
        help = "use DATA_DIR/models/NAME instead of DATA_DIR/model, so one data directory can hold several models (e.g. different splits)",
    )

def run_command(args: argparse.Namespace) -> int:
    from .run import RunConfig, run_problems
    from .wandb_tracking import WandbTracker

    model = run_model(args)
    config = RunConfig(
        mode = GuidanceMode.Base if model is None else GuidanceMode.Weighted,
        policy = args.policy,
        checkpoint = None if model is None else str(model),
        device = args.device,

        seed = args.seed,
        step_limit = args.step_limit,
        timeout_seconds = args.timeout_seconds,
    )

    if config.mode != GuidanceMode.Base:
        from .model import AxiomPredictor

        print_device(args.device)
        assert model is not None
        AxiomPredictor.load(model, device = args.device)

    problems = selected_problems(args)
    log(
        f"axiom-predictor: running {config.policy} in {config.mode} mode on {len(problems)} problems ({json.dumps(config.to_dict(), sort_keys = True)})"
    )

    tracker = WandbTracker.start(
        wandb_config(args),
        job_type = "run",
        run_config = {
            **config.to_dict(),
            "split": selected_split(args).to_dict(),
            "problems": list(problems),
            "cli_arguments": cli_properties(args),
        },
    )

    results: list[dict[str, Any]] = []
    proved = 0
    try:
        for result in run_problems(
            problems,
            tptp_root = args.tptp,
            config = config,
            num_workers = args.num_workers,
        ):
            results.append(result)
            proved += int(bool(result.get("proved")))
            print(json.dumps(result, sort_keys = True), flush = True)
            log(
                f"[{len(results)}/{len(problems)}] {result['problem']}: {result['outcome']} in {float(result.get('seconds') or 0):.2f}s ({proved} proved)"
            )

            if tracker is not None:
                tracker.log_run_result(len(results), len(problems), proved, result)

        times = [float(r["seconds"]) for r in results if r.get("proved")]
        summary = {
            "mode": config.mode,
            "policy": config.policy,
            "problems": len(results),
            "proved": proved,
            "proved_seconds_total": sum(times),
            "proved_seconds_mean": sum(times) / len(times) if times else None,
        }

        log(json.dumps(summary, sort_keys = True))
        if tracker is not None:
            tracker.log_run_results(results, summary)
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code = 1)

        raise

    if tracker is not None:
        tracker.finish()

    return 0

def run_model(args: argparse.Namespace) -> Path | None:
    if args.model is not None:
        if args.model_name is not None:
            raise ValueError("give either --model or --model-name, not both")

        return args.model

    if args.data_dir is None:
        if args.model_name is not None:
            raise ValueError("--model-name needs --data-dir")

        return None

    return model_directory(args.data_dir, args.model_name)

def selected_split(args: argparse.Namespace) -> ProblemSplit:
    parts = tuple(range(args.split)) if args.parts is None else tuple(args.parts)
    return ProblemSplit(args.split, parts)

def selected_problems(args: argparse.Namespace) -> tuple[str, ...]:
    split = selected_split(args)
    problems = (
        expand_problem_inputs(tuple(args.problems), tptp_root = args.tptp)
        if args.problems
        else tptp_problems(tptp_root = args.tptp)
    )

    selected = split.select(problems)
    log(
        f"axiom-predictor: selected {len(selected)} of {len(problems)} problems ({split.describe()})"
    )

    if not selected:
        raise ValueError(f"no problems left after selecting {split.describe()}")

    return selected

def training_request(args: argparse.Namespace) -> tuple[tuple[str, ...], Path | None]:
    if args.dataset is not None and args.problems:
        raise ValueError("train takes either --dataset or PROBLEM inputs, not both")

    if not args.problems:
        return (), selected_dataset(args)

    return selected_problems(args), None

def add_dataset_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset",
        type = Path,
        metavar = "PATH",
        help = "shared dataset directory; collect writes JSONL shards here, train/evaluate read it instead of DATA_DIR/dataset",
    )

def selected_dataset(args: argparse.Namespace) -> Path | None:
    if args.dataset is not None:
        return args.dataset

    return None if args.data_dir is None else args.data_dir / "dataset"

def cli_properties(args: argparse.Namespace) -> dict[str, Any]:
    return {key: jsonable_cli_value(value) for key, value in vars(args).items()}

def jsonable_cli_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [jsonable_cli_value(item) for item in value]

    return value

def add_wandb_arguments(parser: argparse.ArgumentParser):
    toggle = parser.add_mutually_exclusive_group()
    toggle.add_argument(
        "--wandb",
        dest = "wandb",
        action = "store_true",
        help = "attempt W&B logging; warn and continue without it when unavailable",
    )

    toggle.add_argument(
        "--no-wandb",
        dest = "wandb",
        action = "store_false",
        help = "disable W&B logging (tracking otherwise enabled: %(default)s)",
    )

    parser.set_defaults(wandb = True)

def add_split_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--split",
        type = positive_int,
        default = 1,
        metavar = "N",
        help = "partition problems into N parts by a stable hash of their complete filename (including .p)",
    )

    parser.add_argument(
        "--parts",
        "--part",
        dest = "parts",

        type = nonnegative_int,
        nargs = "+",
        metavar = "P",
        help = "use only these parts, each in 0..N-1 (default: every part)",
    )

def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")

    return parsed

def add_worker_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-workers",
        type = positive_int,
        metavar = "N",
        help = (
            "worker processes for collection/search and CPU threads for training/evaluation "
            "(default: maximum available within SLURM_CPUS_PER_TASK and process affinity; capped by problem count for collection/search)"
        ),
    )

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")

    return parsed

def wandb_config(args: argparse.Namespace) -> WandbConfig:
    return WandbConfig(
        enabled = args.wandb,
        name_prefix = wandb_name_prefix(
            getattr(args, "data_dir", None),
            getattr(args, "model_name", None),
        ),
        group = wandb_group(
            getattr(args, "data_dir", None),
            getattr(args, "model_name", None),
        ),
        tags = () if getattr(args, "model_name", None) is None else (args.model_name,),
    )

def wandb_name_prefix(
    data_dir: Path | None,
    model_name: str | None = None,
) -> str | None:
    parts = []
    if data_dir is not None and (
        str(data_dir).startswith("smol") or data_dir.name.startswith("smol")
    ):
        parts.append("smol")

    if model_name is not None:
        parts.append(model_name)

    return "-".join(parts) or None

def wandb_group(data_dir: Path | None, model_name: str | None) -> str | None:
    if model_name is None:
        return None

    return model_name if data_dir is None else f"{data_dir.name}/{model_name}"

if __name__ == "__main__":
    raise SystemExit(main())
