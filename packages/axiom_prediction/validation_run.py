# Guided proof search on a fixed slice of the held-out parts, run after every epoch.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .choices import GuidanceMode, ProverPolicy
from .logs import log
from .run import RunConfig, run_problems
from .split import ProblemSplit
from .tptp import resolve_tptp_problem

def validation_run_configs(policies: list[ProverPolicy], config, checkpoint: Path) -> list[RunConfig]:
    # CPU workers: guided CUDA search would be clamped to a single worker.
    return [
        RunConfig(
            policy = policy,
            checkpoint = str(checkpoint),
            device = "cpu",
            seed = config.seed,
            step_limit = config.step_limit,
            timeout_s = config.timeout_s,
        )
        for policy in policies
    ]

def validation_run_problems(held_out, split: ProblemSplit, ratio: int, tptp_root) -> list[str]:
    problems = split.subsample(sorted({example.problem_path for example in held_out}), ratio)
    if ratio and not problems:
        raise ValueError(f"--validation-run-ratio {ratio} selects no held-out problems")

    for problem in problems:
        resolve_tptp_problem(problem, tptp_root = tptp_root)

    return problems

def run_validation(problems, configs, checkpoint: Path, **options) -> dict[str, float]:
    configs = [replace(config, checkpoint = str(checkpoint)) for config in configs]
    return proof_rates(problems, configs, "run", **options)

# Unguided control on the same problems; it does not change between epochs.
def run_baseline(problems, configs, **options) -> dict[str, float]:
    configs = [replace(config, mode = GuidanceMode.Base, checkpoint = None) for config in configs]
    return proof_rates(problems, configs, "base_run", **options)

def proof_rates(problems, configs, prefix, *, tptp_root, num_workers) -> dict[str, float]:
    metrics = {}
    for config in configs:
        name = f"{prefix}_{config.policy.wire_value()}"
        results = list(run_problems(problems, tptp_root = tptp_root, config = config, num_workers = num_workers))
        proved = sum(bool(result.get("proved")) for result in results)
        errors = sum(bool(result.get("error")) for result in results)
        log(f"validation {name}: {proved}/{len(problems)} proved, {errors} errors")
        metrics[name] = proved / len(problems)

    return metrics
