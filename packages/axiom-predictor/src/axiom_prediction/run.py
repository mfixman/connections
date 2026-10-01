from __future__ import annotations

from .choices import GuidanceMode, ProverPolicy, plain_values

from typing import Any

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import math
from pathlib import Path
import time

from connections.parsing.tptp import TPTPParseError
from connections.agent.sat import SATCoPCon, SATResetCoP
from connections.interaction.szs import SUCCESS
from connections.interaction.run import Problem, run_schedule
from connections.interaction.strategy import MatrixOptions, PolicyOptions, Strategy, StrategySchedule

from .search_workers import supervised_results
from .limits import CollectionTimeout, wall_clock
from .parallel import determine_worker_count
from .tptp import DEFAULT_STEP_LIMIT, DEFAULT_TIMEOUT_SECONDS, load_tptp_problem, resolve_tptp_problem
from .tptp import declared_tptp_status
from .graph import UnsupportedAxiomProblem

RUN_MODES = tuple(GuidanceMode)
RUN_POLICIES = tuple(ProverPolicy)
_NON_REFUTABLE = {
    "satisfiable": "DeclaredSatisfiable",
    "countersatisfiable": "DeclaredCounterSatisfiable",
}

@dataclass(frozen = True, slots = True)
class RunConfig:
    mode: GuidanceMode | str = GuidanceMode.Weighted
    policy: ProverPolicy | str | None = None
    checkpoint: str | None = None
    device: str = "cuda"
    multiprocess: bool = False
    inference_address: tuple[str, int] | None = field(default=None, repr=False)
    inference_key: str | None = field(default=None, repr=False)

    temperature: float = 1.0
    top_k: int | None = None

    seed: int = 0
    step_limit: int = DEFAULT_STEP_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self):
        object.__setattr__(self, "mode", GuidanceMode(self.mode))
        if self.policy is not None:
            object.__setattr__(self, "policy", ProverPolicy(self.policy))

        if self.mode != GuidanceMode.Base and self.checkpoint is None:
            raise ValueError(f"--mode {self.mode} needs --model")

        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("--temperature must be finite and positive")

        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be finite and positive")

        if self.step_limit < 0:
            raise ValueError("--step-limit must be nonnegative")

        if self.top_k is not None and self.top_k < 1:
            raise ValueError("--top-k must be at least 1")

        if self.mode == GuidanceMode.Base and self.top_k is not None:
            raise ValueError("--top-k needs a model; it cannot be used with --mode base")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("inference_address")
        values.pop("inference_key")
        return plain_values(values)

def run_problem(
    problem: str,
    *,
    tptp_root: str | Path | None,
    config: RunConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with wall_clock(config.timeout_seconds):
            result = search_problem(problem, tptp_root = tptp_root, config = config)
    except CollectionTimeout:
        result = {"problem": problem, "outcome": "Timeout", "proved": False}

    result["seconds"] = time.monotonic() - started
    return result

def search_problem(problem, *, tptp_root, config):
    if config.inference_address is not None:
        from .multiprocess import shared_predictor
        predictor = shared_predictor(config.inference_address, config.inference_key)
    else:
        predictor = None if config.mode == GuidanceMode.Base else cached_predictor(
            config.checkpoint, config.device,
        )

    from .training import label_policy

    metadata = {} if predictor is None else predictor.training_config
    policy = label_policy(config.policy, {"collection": metadata})
    config = replace(config, policy = policy)
    out: dict[str, Any] = {
        "problem": problem,
        "mode": GuidanceMode(config.mode).wire_value,
        "policy": ProverPolicy(config.policy).wire_value,
        "seed": config.seed,
    }

    started = time.monotonic()
    try:
        path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
        declared = declared_tptp_status(path)
        skip = None if declared is None else _NON_REFUTABLE.get(declared.casefold())
        if skip is not None:
            out.update(outcome = skip, proved = False, seconds = 0.0)
            return out

        policy_class = {
            ProverPolicy.SatResetCoP: SATResetCoP,
            ProverPolicy.SatCoP: SATCoPCon,
        }[config.policy]

        args: dict[str, Any] = {"seed": config.seed}
        if predictor is not None:
            try:
                loaded = load_tptp_problem(problem, tptp_root = tptp_root)
            except UnsupportedAxiomProblem as error:
                predictor = None
                out["guidance_fallback"] = str(error)

        if predictor is not None:
            from .guided import AxiomGuidedSATCoP, AxiomGuidedSATResetCoP, matrix_digest

            predictions = predictor.predict(
                loaded.matrix,
                axiom_clause_ids = loaded.axiom_clause_ids,
                conjecture_clause_ids = loaded.conjecture_clause_ids,
            )

            weights = {p.clause_index: p.probability for p in predictions}
            weights.update({i: 1.0 for i in loaded.conjecture_clause_ids})
            allowed = None
            if config.top_k is not None:
                allowed = tuple(
                    sorted(
                        {p.clause_index for p in predictions if p.rank <= config.top_k}
                        | set(loaded.conjecture_clause_ids)
                    )
                )

            policy_class = {
                ProverPolicy.SatResetCoP: AxiomGuidedSATResetCoP,
                ProverPolicy.SatCoP: AxiomGuidedSATCoP,
            }[config.policy]

            args.update(
                clause_weights = weights,
                mode = config.mode,
                temperature = config.temperature,
                allowed_clause_ids = allowed,
                matrix_digest = matrix_digest(loaded.matrix),
            )

            out.update(
                axioms = len(loaded.axiom_clause_ids),
                kept_axioms = len(loaded.axiom_clause_ids)
                if allowed is None
                else len(allowed) - len(loaded.conjecture_clause_ids),
            )

            del loaded, predictions

        prediction_seconds = time.monotonic() - started
        remaining = config.timeout_seconds - prediction_seconds
        if remaining <= 0:
            out.update(
                outcome = "Timeout",
                proved = False,
                seconds = prediction_seconds,
                prediction_seconds = prediction_seconds,
            )

            return out

        result = run_schedule(
            Problem(
                path,
                logic = "classical",
                domain = "constant",
                source_file_dirs = () if root is None else (root,),
            ),
            schedule = StrategySchedule.single(
                Strategy(
                    matrix = MatrixOptions(mark_conjecture = True),
                    policy = PolicyOptions(policy_class = policy_class, args = args),
                ),
                steps = config.step_limit,
                timeout_seconds = remaining,
            ),
        )

        strategy = result.strategy_results[0] if result.strategy_results else None
        out.update(
            outcome = "unknown" if result.szs_status is None else result.szs_status.value,
            proved = result.szs_status in SUCCESS,
            steps = None if strategy is None else strategy.steps,
            proof_size = None if strategy is None else strategy.proof_size,
            seconds = time.monotonic() - started,
        )

        if predictor is not None:
            out["prediction_seconds"] = prediction_seconds
    except TPTPParseError as error:
        out.update(
            outcome = f"{type(error).__name__}: {' '.join(str(error).split())}",
            proved = False,
            parseable = False,
            seconds = time.monotonic() - started,
        )
    except Exception as error:
        out.update(
            outcome = f"{type(error).__name__}: {' '.join(str(error).split())}",
            proved = False,
            error = True,
            seconds = time.monotonic() - started,
        )

    return out

def run_problems(
    problems: list[str] | tuple[str, ...],
    *,
    tptp_root: str | Path | None,
    config: RunConfig,
    num_workers: int | None = None,
) -> Iterator[dict[str, Any]]:
    workers = determine_worker_count(len(problems), num_workers)
    if config.multiprocess and config.mode != GuidanceMode.Base and workers:
        from .multiprocess import inference_service
        with inference_service(config.checkpoint, config.device, workers) as (address, key):
            shared = replace(config, inference_address=address, inference_key=key)
            yield from supervised_results(
                run_one, ((p, tptp_root, shared) for p in problems),
                workers=workers, timeout=config.timeout_seconds,
            )
        return

    if config.mode != GuidanceMode.Base and num_workers is None:
        from .model import resolve_device

        if resolve_device(config.device).type == "cuda":
            workers = min(workers, 1)

    if workers == 0:
        return

    yield from supervised_results(
        run_one,
        ((p, tptp_root, config) for p in problems),
        workers = workers,
        timeout = config.timeout_seconds,
    )

def run_one(
    problem: str,
    tptp_root: str | Path | None,
    config: RunConfig,
) -> dict[str, Any]:
    init_worker(config.mode != GuidanceMode.Base)
    return run_problem(problem, tptp_root = tptp_root, config = config)

def init_worker(uses_model: bool):
    if not uses_model:
        return

    try:
        import torch
    except ImportError:
        return

    torch.set_num_threads(1)

@lru_cache(maxsize = 2)
def cached_predictor(checkpoint: str | None, device: str):
    from .model import AxiomPredictor

    if checkpoint is None:
        raise ValueError("guided search requires a checkpoint")

    return AxiomPredictor.load(checkpoint, device = device)

__all__ = ["RUN_MODES", "RUN_POLICIES", "RunConfig", "run_problem", "run_problems"]
