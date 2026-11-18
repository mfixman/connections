# `run`: proof search on each problem in a worker process, unguided or with clause weights from a model or SInE.

from typing import Any

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import time

from connections.agent.sat import SATCoPCon, SATResetCoP
from connections.interaction.szs import SUCCESS as success
from connections.interaction.run import Problem, run_schedule
from connections.interaction.strategy import MatrixOptions, PolicyOptions, Strategy, StrategySchedule

from .choices import GuidanceMode, ProverPolicy, plain_values
from .limits import CollectionTimeout, default_step_limit, default_timeout_s, wall_clock
from .parallel import determine_worker_count
from .search_workers import supervised_results
from .tptp import declared_non_refutable, load_tptp_problem, resolve_tptp_problem, selector_equality_axioms

@dataclass(slots = True, init = False)
class RunConfig:
    mode: GuidanceMode
    policy: ProverPolicy
    checkpoint: str | None
    device: str
    temperature: float
    seed: int
    step_limit: int
    timeout_s: float

    def __init__(
        self,
        mode = GuidanceMode.Weighted,
        policy = ProverPolicy.SatResetCoP,
        checkpoint = None,
        device = "cuda",
        temperature = 1.0,
        seed = 0,
        step_limit = default_step_limit,
        timeout_s = default_timeout_s,
    ):
        self.mode = GuidanceMode(mode)
        self.policy = ProverPolicy(policy)
        self.checkpoint = checkpoint
        self.device = device
        self.temperature = 1.0 if self.mode == GuidanceMode.Sine else temperature
        self.seed = seed
        self.step_limit = step_limit
        self.timeout_s = timeout_s

        if (self.mode == GuidanceMode.Weighted) != (checkpoint is not None):
            raise ValueError(f"{self.mode} mode and checkpoint {checkpoint} do not go together")

    def uses_model(self):
        return self.mode == GuidanceMode.Weighted

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["timeout_seconds"] = values.pop("timeout_s")
        if self.mode == GuidanceMode.Sine:
            from .sine import settings

            values.update(settings())

        return plain_values(values)

def run_problem(
    problem: str,
    *,
    tptp_root: str | Path | None,
    config: RunConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with wall_clock(config.timeout_s):
            result = search_problem(problem, tptp_root = tptp_root, config = config)
    except CollectionTimeout:
        result = {"problem": problem, "outcome": "Timeout", "proved": False}

    result["seconds"] = time.monotonic() - started
    return with_run_metadata(result, config)

def search_problem(problem, *, tptp_root, config):
    predictor = cached_predictor(config.checkpoint, config.device) if config.uses_model() else None
    out: dict[str, Any] = {
        "problem": problem,
        "mode": config.mode.wire_value(),
        "policy": config.policy.wire_value(),
        "seed": config.seed,
    }

    started = time.monotonic()
    try:
        path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
        skip = declared_non_refutable(path)
        if skip is not None:
            out.update(outcome = skip, proved = False, seconds = 0.0)
            return out

        if config.policy == ProverPolicy.SatCoP:
            from .satcop_search import search_native

            out.update(search_native(path, root, config, predictor, started))
            return out

        policy_class = {
            ProverPolicy.SatResetCoP: SATResetCoP,
            ProverPolicy.SatCoPCon: SATCoPCon,
        }[config.policy]

        args: dict[str, Any] = {"seed": config.seed}
        guided = config.mode != GuidanceMode.Base
        if guided:
            loaded = load_tptp_problem(problem, tptp_root = tptp_root)
            from .guided import AxiomGuidedSATCoP, AxiomGuidedSATResetCoP, clause_weights, matrix_digest

            weights = clause_weights(
                loaded.matrix,
                loaded.axiom_clause_ids,
                loaded.conjecture_clause_ids,
                predictor,
            )
            policy_class = {
                ProverPolicy.SatResetCoP: AxiomGuidedSATResetCoP,
                ProverPolicy.SatCoPCon: AxiomGuidedSATCoP,
            }[config.policy]

            args.update(
                clause_weights = weights,
                temperature = config.temperature,
                matrix_digest = matrix_digest(loaded.matrix),
            )

            out["axioms"] = len(loaded.axiom_clause_ids)

            del loaded

        prediction_s = time.monotonic() - started
        remaining = config.timeout_s - prediction_s
        if remaining <= 0:
            out.update(
                outcome = "Timeout",
                proved = False,
                seconds = prediction_s,
                prediction_seconds = prediction_s,
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
                    matrix = MatrixOptions(mark_conjecture = True, equality_axioms = selector_equality_axioms()),
                    policy = PolicyOptions(policy_class = policy_class, args = args),
                ),
                steps = config.step_limit,
                timeout_seconds = remaining,
            ),
        )

        strategy = result.strategy_results[0] if result.strategy_results else None
        out.update(
            outcome = "unknown" if result.szs_status is None else result.szs_status.value,
            proved = result.szs_status in success,
            steps = None if strategy is None else strategy.steps,
            proof_size = None if strategy is None else strategy.proof_size,
            seconds = time.monotonic() - started,
        )

        if guided:
            out["prediction_seconds"] = prediction_s
    except Exception as error:
        out.update(
            outcome = f"{type(error).__name__}: {' '.join(str(error).split())}",
            proved = False,
            error = True,
            seconds = time.monotonic() - started,
        )

    return out

def run_problems(problems, *, tptp_root, config, num_workers = None) -> Iterator[dict[str, Any]]:
    if config.policy == ProverPolicy.SatCoP:
        from .satcop_search import search_binary

        search_binary()

    workers = determine_worker_count(len(problems), num_workers)
    # One model per worker does not fit on a GPU, so guided CUDA search runs alone unless told otherwise.
    if config.uses_model() and num_workers is None:
        from .model import resolve_device

        if resolve_device(config.device).type == "cuda":
            workers = min(workers, 1)

    if workers == 0:
        return

    requests = ((problem, tptp_root, config) for problem in problems)
    for result in supervised_results(run_one, requests, workers = workers, timeout = config.timeout_s):
        yield with_run_metadata(result, config)

def with_run_metadata(result, config):
    result.setdefault("error", False)
    result.setdefault("seed", config.seed)
    result.setdefault("mode", config.mode.wire_value())
    result.setdefault("policy", config.policy.wire_value())
    return result

def run_one(problem, tptp_root, config):
    if config.uses_model():
        import torch

        torch.set_num_threads(1)

    return run_problem(problem, tptp_root = tptp_root, config = config)

@lru_cache(maxsize = 2)
def cached_predictor(checkpoint, device):
    from .model import AxiomPredictor

    return AxiomPredictor.load(checkpoint, device = device)
