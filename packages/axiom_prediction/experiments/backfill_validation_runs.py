# Backfill the per-epoch validation runs of an earlier training run into its W&B run.

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re

from axiom_prediction.choices import ProverPolicy
from axiom_prediction.data import axiom_example_schema
from axiom_prediction.dataset import collection_caches, read_dataset_rows
from axiom_prediction.logs import log
from axiom_prediction.split import ProblemSplit
from axiom_prediction.tptp import resolve_tptp_problem
from axiom_prediction.validation_run import run_baseline, run_validation, validation_run_configs
from axiom_prediction.wandb_tracking import WandbConfig

def parse_arguments():
    parser = argparse.ArgumentParser(description = "backfill per-epoch validation runs into a W&B training run")
    parser.add_argument("--run-id", required = True)
    parser.add_argument("--output", type = Path, required = True)
    parser.add_argument("--ratio", type = int, default = 10)
    parser.add_argument("--provers", type = ProverPolicy, nargs = "+", default = [ProverPolicy.SatResetCoP])
    parser.add_argument("--epoch-jobs", type = int, default = 4)
    parser.add_argument("--num-workers", type = int, default = 19)
    parser.add_argument("--dry-run", action = "store_true")
    parser.add_argument("--skip-upload", action = "store_true")
    parser.add_argument("--replace-from-step", type = int, metavar = "STEP")
    return parser.parse_args()

def wandb_key():
    key = os.environ.get("WANDB_API_KEY", "").strip()
    return key or WandbConfig().key_file.expanduser().read_text(encoding = "utf-8").strip()

def training_run(run_id):
    import wandb

    wandb_config = WandbConfig()
    return wandb.Api(api_key = wandb_key()).run(f"{wandb_config.entity}/{wandb_config.project}/{run_id}")

# Reads only the leading columns of cache lines, skipping the graphs that training would decode.
def shard_paths(shard, max_bytes):
    if not (shard / "cache.csv").exists():
        rows = read_dataset_rows(shard, max_bytes)
        return {row["problem_path"] for row in rows if row.get("schema") == axiom_example_schema}

    example = re.compile(r'^"""([^"]+)""",,,,"""([^"]+)"""')
    paths = set()
    with (shard / "cache.csv").open(encoding = "utf-8", newline = "") as stream:
        for line in stream:
            match = example.match(line)
            kept = line.endswith("\n") and len(line) <= max_bytes
            if kept and match and match[1] == axiom_example_schema:
                paths.add(match[2])

    return paths

def held_out_paths(dataset, split, max_bytes):
    # Collection shards used the same split, so only the held-out parts' shards hold their problems.
    shards = [
        shard
        for shard in collection_caches(dataset)
        if any(shard.name.endswith(f"-p{part}") for part in split.parts)
    ]

    paths = set()
    for shard in shards:
        paths |= shard_paths(shard, max_bytes)

    outside = {path for path in paths if not split.contains(path)}
    if outside:
        raise ValueError(f"{len(outside)} problems in held-out shards are outside {split.describe()}")

    log(f"{len(paths)} held-out problems with proofs in {len(shards)} shards of {dataset}")
    return paths

# Saved on the first attempt, since the dataset can grow while collections still run.
def validation_selection(settings, ratio, output):
    path = output / "selection.json"
    if path.exists():
        return json.loads(path.read_text())

    split = ProblemSplit(**settings["evaluation_split"])
    max_bytes = int(settings["cli_arguments"]["max_example_mb"] * 1e6)
    paths = held_out_paths(Path(settings["dataset"]), split, max_bytes)
    problems = split.subsample(sorted(paths), ratio)
    for problem in problems:
        resolve_tptp_problem(problem, tptp_root = settings["cli_arguments"].get("tptp"))

    selection = {"held_out": len(paths), "problems": problems}
    path.write_text(json.dumps(selection, indent = 2) + "\n")
    return selection

def read_results(path):
    if not path.exists():
        return {}

    with path.open(newline = "") as stream:
        return {(int(row["epoch"]), row["metric"]): float(row["value"]) for row in csv.DictReader(stream)}

def append_results(path, epoch, metrics):
    new = not path.exists()
    with path.open("a", newline = "") as stream:
        writer = csv.DictWriter(stream, fieldnames = ["epoch", "metric", "value"])
        if new:
            writer.writeheader()

        writer.writerows({"epoch": epoch, "metric": name, "value": value} for name, value in metrics.items())

def epoch_checkpoints(model_dir):
    return {int(path.stem.split("-")[1]): path for path in sorted(model_dir.glob("epoch-*.ts"))}

# Epoch 0 holds the unguided baseline.
def evaluate_epochs(problems, configs, checkpoints, path, args, tptp_root):
    done = {epoch for epoch, _ in read_results(path)}
    options = dict(tptp_root = tptp_root, num_workers = args.num_workers)
    with ThreadPoolExecutor(args.epoch_jobs) as pool:
        futures = {}
        if 0 not in done:
            futures[pool.submit(run_baseline, problems, configs, **options)] = 0

        for epoch, checkpoint in checkpoints.items():
            if epoch not in done:
                futures[pool.submit(run_validation, problems, configs, checkpoint, **options)] = epoch

        for future in as_completed(futures):
            epoch = futures[future]
            metrics = future.result()
            append_results(path, epoch, metrics)
            log(f"validation backfill: epoch {epoch} recorded {metrics}")

# Epoch 0 is the unguided baseline, repeated on every guided epoch.
def epoch_metrics(results, epoch, policies, selection):
    tested = len(selection["problems"])
    metrics = {name: value for (e, name), value in results.items() if e in (0, epoch)}
    for policy in policies:
        solved = metrics[f"run_{policy}"]
        metrics[f"test_{policy}_num"] = tested
        metrics[f"test_{policy}_ratio"] = tested / selection["held_out"]
        metrics[f"test_{policy}_solved_num"] = round(solved * tested)
        metrics[f"test_{policy}_solved_ratio"] = solved

    return metrics

# Rewinding archives the history from that step on, replacing an earlier backfill.
def upload(run_id, rows, details, output, replace_step = None):
    import wandb

    if training_run(run_id).state == "running":
        raise RuntimeError(f"W&B run {run_id} is running elsewhere; results kept in {output}")

    wandb_config = WandbConfig()
    wandb.login(key = wandb_key())
    target = dict(id = run_id, resume = "must")
    if replace_step is not None:
        target = dict(resume_from = f"{run_id}?_step={replace_step}")

    run = wandb.init(
        entity = wandb_config.entity,
        project = wandb_config.project,
        dir = str(output),
        **target,
    )

    run.define_metric("epoch")
    run.define_metric("evaluation/*", step_metric = "epoch")
    for epoch, metrics in sorted(rows.items()):
        run.log({"epoch": epoch, **{f"evaluation/{name}": value for name, value in metrics.items()}})

    run.config.update({"validation_backfill": details}, allow_val_change = True)
    run.finish()

def main():
    args = parse_arguments()
    args.output.mkdir(parents = True, exist_ok = True)
    settings = training_run(args.run_id).config
    selection = validation_selection(settings, args.ratio, args.output)
    problems = selection["problems"]
    log(f"validation backfill: {len(problems)} problems (1/{args.ratio} of {settings['evaluation_split']})")
    if args.dry_run:
        return

    model_dir = Path(settings["cli_arguments"]["data_dir"]) / "model"
    checkpoints = epoch_checkpoints(model_dir)
    limits = argparse.Namespace(
        seed = settings["seed"],
        step_limit = settings["step_limit"],
        timeout_s = settings["timeout_seconds"],
    )

    configs = validation_run_configs(args.provers, limits, model_dir / "model.ts")

    path = args.output / "results.csv"
    evaluate_epochs(problems, configs, checkpoints, path, args, settings["cli_arguments"].get("tptp"))
    details = {
        "ratio": args.ratio,
        "problems": len(problems),
        "held_out": selection["held_out"],
        "split": settings["evaluation_split"],
        "provers": [config.policy.wire_value() for config in configs],
        "epochs": max(checkpoints),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }

    if args.skip_upload:
        return

    if (args.output / "uploaded.json").exists():
        log(f"validation backfill: already uploaded to W&B run {args.run_id}")
        return

    results = read_results(path)
    rows = {
        epoch: epoch_metrics(results, epoch, details["provers"], selection)
        for epoch in checkpoints
    }

    upload(args.run_id, rows, details, args.output, args.replace_from_step)
    (args.output / "uploaded.json").write_text(json.dumps(details, indent = 2) + "\n")
    log(f"validation backfill: uploaded {len(rows)} epochs to W&B run {args.run_id}")

if __name__ == "__main__":
    main()
