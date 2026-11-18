# Command-line interface: collect SAT-core labels, join them into a dataset, train a network, run guided proof search.

import argparse
import os
from pathlib import Path

from .choices import LossWeighting, ProverPolicy
from .limits import default_collection_timeout_s, default_timeout_s
from .logs import log, monitor_progress
from .models import available_models
from .output import output_format, write_record
from .split import ProblemSplit, split_scheme

def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")

    return parsed

def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")

    return parsed

def add_split_arguments(parser):
    parser.add_argument("--split-scheme", type = int, choices = (3, 4, 5), default = split_scheme, help = "split algorithm")
    parser.add_argument("--split", type = positive_int, default = 1, metavar = "N", help = "partition problems into N parts")
    parser.add_argument(
        "--parts", "--part", type = nonnegative_int, nargs = "+", metavar = "P",
        help = "use only these parts, each in 0..N-1 (default: every part)",
    )

def add_device_argument(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--device", default = "cuda", help = "cuda or cuda:N needs a GPU; auto falls back to the CPU")
    group.add_argument("--cpu", dest = "device", action = "store_const", const = "cpu", help = "same as --device cpu")

def add_wandb_arguments(parser):
    parser.add_argument("--run-name", help = "W&B run name")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "log to W&B if its key is available")
    parser.add_argument("--no-wandb", dest = "wandb", action = "store_false")

def add_common_arguments(parser):
    parser.add_argument("--tptp", type = Path, help = "TPTP root (default: $TPTP, then TPTP directories near the checkout)")
    parser.add_argument("--num-workers", type = positive_int, metavar = "N", help = "processes (default: the job's CPUs)")
    parser.add_argument("--equality-axioms", action = "store_true", help = "add explicit equality axioms to the matrix")
    add_split_arguments(parser)

def add_collect_parser(subparsers):
    problems_help = "files, directories or TPTP categories (default: all of TPTP)"
    policies = list(ProverPolicy)

    collect = subparsers.add_parser("collect", help = "prove problems and append their SAT-core labels to problems.csv")
    collect.add_argument("problems", metavar = "PROBLEM", nargs = "*", help = problems_help)
    collect.add_argument("--problems-from", type = Path, metavar = "FILE", help = "also the problems listed in FILE")
    collect.add_argument("--dataset", type = Path, required = True, help = "directory of problems.csv; resumed if it exists")

    collect.add_argument("--policy", type = ProverPolicy, choices = policies, default = ProverPolicy.SatResetCoP)
    collect.add_argument("--seed", type = nonnegative_int, default = 0, help = "proof-search seed")
    collect.add_argument("--timeout-seconds", dest = "timeout_s", type = float, default = default_collection_timeout_s)
    collect.add_argument("--satcop-location", type = Path, help = "native SATCoP checkout or executable (default: ~/TP/satcop)")
    add_common_arguments(collect)

def add_join_parser(subparsers):
    join = subparsers.add_parser("join", help = "join collect results and joined datasets into one deduplicated dataset")
    join.add_argument("inputs", metavar = "INPUT", nargs = "+", type = Path, help = "collect results or joined datasets")
    join.add_argument("--dataset", type = Path, required = True, help = "directory receiving data.npz and metadata.json")
    join.add_argument("--num-workers", type = positive_int, metavar = "N")

def add_train_parser(subparsers):
    weighting = dict(type = LossWeighting, choices = list(LossWeighting), default = LossWeighting.Clause)
    provers = dict(type = ProverPolicy, choices = list(ProverPolicy), nargs = "+", default = [ProverPolicy.SatResetCoP])
    weighting_help = "average the loss over all clauses of a batch, or over problems whose cores share one weight"
    ratio_help = "run guided proof search on 1/R of the --evaluate problems after each epoch (0 disables)"

    train = subparsers.add_parser("train", help = "train a network on a joined dataset")
    train.add_argument("--data-dir", type = Path, required = True, help = "run directory; checkpoints go to DATA_DIR/model")
    train.add_argument("--dataset", type = Path, required = True, help = "joined dataset")
    train.add_argument("--resume", action = "store_true", help = "continue from DATA_DIR/model up to --epochs in total")

    train.add_argument("--network", default = "DefaultFull", choices = available_models())
    train.add_argument("--epochs", type = positive_int, default = 200)
    train.add_argument("--batch-size", type = positive_int, default = 43, help = "problems per optimizer update")
    train.add_argument("--seed", type = int, default = 0)

    train.add_argument("--max-example-mb", type = float, help = "skip examples whose collected row was larger")
    train.add_argument("--min-axioms", type = nonnegative_int, default = 0, help = "skip examples with at most N axioms")
    train.add_argument("--loss-weighting", help = weighting_help, **weighting)

    train.add_argument("--evaluate", type = nonnegative_int, nargs = "+", metavar = "PART", help = "held-out parts to score")
    train.add_argument("--evaluate-every", type = positive_int, default = 1, metavar = "EPOCHS")
    train.add_argument("--validation-run-ratio", type = nonnegative_int, default = 10, metavar = "R", help = ratio_help)
    train.add_argument("--validation-run-provers", metavar = "POLICY", **provers)
    train.add_argument("--validation-all-held-out", action = "store_true", help = "validate on unlabelled held-out problems too")
    train.add_argument("--timeout-seconds", dest = "timeout_s", type = float, default = default_timeout_s, help = "per validation search")

    add_device_argument(train)
    add_wandb_arguments(train)
    add_common_arguments(train)

def add_run_parser(subparsers):
    problems_help = "files, directories or TPTP categories (default: all of TPTP)"
    policy_help = "default: the policy that labelled the checkpoint's data, or SatResetCoP"

    run = subparsers.add_parser("run", help = "proof search, unguided or guided by a checkpoint or SInE")
    run.add_argument("problems", metavar = "PROBLEM", nargs = "*", help = problems_help)
    source = run.add_mutually_exclusive_group()
    source.add_argument("--model", type = Path, help = "checkpoint: a .ts file, a legacy .pt file or their directory")
    source.add_argument("--sine", action = "store_true", help = "weigh clauses by clause-level SInE instead of a model")

    run.add_argument("--policy", type = ProverPolicy, choices = list(ProverPolicy), help = policy_help)
    run.add_argument("--seed", type = int, default = 0)
    run.add_argument("--timeout-seconds", dest = "timeout_s", type = float, default = default_timeout_s)
    run.add_argument("--temperature", type = float, default = 1.0, help = "clause weights become probability ** (1 / T)")
    run.add_argument("--output", type = Path, help = "write the CSV rows to this file and resume from it")

    add_device_argument(run)
    add_wandb_arguments(run)
    add_common_arguments(run)

def build_parser():
    parser = argparse.ArgumentParser(prog = "axiom-predictor")
    subparsers = parser.add_subparsers(dest = "command", required = True)
    for add_parser in (add_collect_parser, add_join_parser, add_train_parser, add_run_parser):
        add_parser(subparsers)

    for command in subparsers.choices.values():
        command.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    return parser

def selected_split(args):
    parts = list(range(args.split)) if args.parts is None else list(args.parts)
    return ProblemSplit(args.split, parts, scheme = args.split_scheme)

def selected_problems(args):
    from .tptp import expand_problem_inputs, tptp_problems

    split = selected_split(args)
    if args.problems:
        problems = expand_problem_inputs(args.problems, tptp_root = args.tptp)
    else:
        problems = tptp_problems(tptp_root = args.tptp)

    selected = split.select(problems)
    log(f"selected {len(selected)} of {len(problems)} problems ({split.describe()})")
    if not selected:
        raise ValueError(f"no problems left after selecting {split.describe()}")

    return selected

def wandb_config(args):
    from .wandb_tracking import WandbConfig

    return WandbConfig(enabled = args.wandb, name = args.run_name)

# Saved in the W&B run, where the backfill scripts read it.
def cli_properties(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

def collect_command(args):
    from .collection import collect_axiom_dataset

    if args.problems_from is not None:
        listed = list(filter(None, map(str.strip, args.problems_from.read_text().splitlines())))
        if not listed:
            raise ValueError(f"no problems listed in {args.problems_from}")

        args.problems = [*args.problems, *listed]

    summary = collect_axiom_dataset(
        selected_problems(args),
        output_dir = args.dataset,
        tptp_root = args.tptp,
        timeout_s = args.timeout_s,

        sat_policy = args.policy,
        num_workers = args.num_workers,
        satcop_location = args.satcop_location,
        seed = args.seed,
    )

    write_record(summary)
    return 0

def join_command(args):
    from .joined import join_datasets

    write_record(join_datasets(args.inputs, args.dataset, args.num_workers))
    return 0

def train_command(args):
    from .model import device_description
    from .training import AxiomTrainingConfig, train_axiom_predictor

    log(f"compute device: {device_description(args.device)}")
    evaluation_split = None
    if args.evaluate is not None:
        evaluation_split = ProblemSplit(args.split, list(args.evaluate), scheme = args.split_scheme)

    config = AxiomTrainingConfig(
        model = args.network,
        epochs = args.epochs,
        device = args.device,
        seed = args.seed,

        batch_size = args.batch_size,
        timeout_s = args.timeout_s,
        num_workers = args.num_workers,
    )

    args.data_dir.mkdir(parents = True, exist_ok = True)
    train_axiom_predictor(
        dataset = args.dataset,
        output_dir = args.data_dir / "model",
        tptp_root = args.tptp,
        config = config,
        wandb_config = wandb_config(args),
        run_properties = cli_properties(args),

        split = selected_split(args),
        evaluation_split = evaluation_split,
        evaluate_every = args.evaluate_every,
        validation_run_ratio = args.validation_run_ratio,
        validation_run_provers = args.validation_run_provers,
        resume = args.resume,

        max_example_mb = args.max_example_mb,
        loss_weighting = args.loss_weighting,
        min_axioms = args.min_axioms,
        validation_all_held_out = args.validation_all_held_out,
    )

    return 0

def run_command(args):
    from .choices import GuidanceMode
    from .dataset import label_policy
    from .output import journal, report_metrics
    from .run import RunConfig, run_problems
    from .wandb_tracking import WandbTracker

    network_size, labelled = 0, {}
    if args.model is not None:
        from .model import AxiomPredictor

        predictor = AxiomPredictor.load(args.model, device = "cpu")
        network_size, labelled = predictor.network_size(), {"collection": predictor.training_config}

    config = RunConfig(
        mode = GuidanceMode.Sine if args.sine else GuidanceMode.Base if args.model is None else GuidanceMode.Weighted,
        policy = label_policy(args.policy, labelled),
        checkpoint = None if args.model is None else str(args.model),
        device = args.device,

        seed = args.seed,
        timeout_s = args.timeout_s,
        temperature = args.temperature,
    )

    split, properties = selected_split(args), cli_properties(args)
    problems = selected_problems(args)
    log(f"running {len(problems)} problems with {config.to_dict()}")
    session = journal.get()

    results = [] if session is None else [row for row in session.records if row.get("problem")]
    completed = {row["problem"] for row in results}
    if completed - set(problems):
        raise ValueError("output contains problems outside this selection; use a new --output")

    tracker = WandbTracker.start(
        wandb_config(args),
        job_type = "run",
        run_config = {**config.to_dict(), "split": split.to_dict(), "problems": problems, "cli_arguments": properties},
        output_dir = None if session is None else session.directory,
        resume = session is not None,
    )

    try:
        pending = [problem for problem in problems if problem not in completed]
        for result in run_problems(pending, tptp_root = args.tptp, config = config, num_workers = args.num_workers):
            result["part"] = split.part(result["problem"])
            result["network_size"] = network_size
            results.append(result)
            write_record(result)

            proved = sum(bool(row.get("proved")) for row in results)
            progress = f"[{len(results)}/{len(problems)}] {result['problem']}: {result['outcome']}"
            log(f"{progress} in {result['seconds']:.2f}s ({proved} proved)")
            if tracker is not None:
                tracker.log_run_result(len(results), len(problems), proved, result)

        times = [float(row["seconds"]) for row in results if row.get("proved")]
        summary = {
            "mode": config.mode,
            "policy": config.policy,
            "problems": len(results),
            "proved": len(times),

            "ratio_problems_proved": len(times) / len(results) if results else None,
            "proved_seconds_total": sum(times),
            "proved_seconds_mean": sum(times) / len(times) if times else None,
        }

        report_metrics(summary)
        if tracker is not None:
            tracker.log_run_results(results, summary)
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code = 1)

        raise

    if tracker is not None:
        tracker.finish()

    return 0 if any(not result.get("error") for result in results) else 2

@monitor_progress()
def main(argv = None):
    from .collection import NoParseableProblemsError
    from .tptp import NoProblemFilesError

    commands = {"collect": collect_command, "join": join_command, "train": train_command, "run": run_command}
    args = build_parser().parse_args(argv)
    # The choice travels in the environment so that worker processes see it too.
    os.environ["axiom_selector_equality_axioms"] = "1" if getattr(args, "equality_axioms", False) else "0"
    log(f"axiom-predictor: starting {args.command}")
    try:
        if args.command == "train":
            return train_command(args)

        with output_format(args.command, getattr(args, "output", None), cli_properties(args)) as session:
            if session is not None and session.complete():
                log("output is already complete; nothing to resume")
                return session.exit_code()

            exit_code = commands[args.command](args)
            if session is not None:
                session.finish(exit_code)

            return exit_code
    except (NoProblemFilesError, NoParseableProblemsError) as error:
        log(f"axiom-predictor: error: {error}")
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        log(f"axiom-predictor: error: {error}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
