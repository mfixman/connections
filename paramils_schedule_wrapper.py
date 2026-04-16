"""ParamILS wrapper for static MyPyCop strategy schedules.

This script exposes a static schedule of MyPyCop strategies as a single
"algorithm" that ParamILS can optimize.

ParamILS calls wrappers as:

    algo_executable <instance> <run_obj> <cutoff_time> <cutoff_length> <seed> <params...>

and expects a line of the form:

    Result for ParamILS: solved, runtime, runlength, best_sol, seed

The wrapper below accepts three schedule slots. Each non-final active slot gets
some percentage of the *remaining* budget. The final active slot gets whatever
budget remains. This avoids having to encode sum-to-budget constraints in the
ParamILS parameter file.

Example usage from the shell:

    python paramils_schedule_wrapper.py SYN000+1.p runtime 60 100000 -1 \\
        slot1_strategy=LeanCopCut slot1_share=60 \\
        slot2_strategy=HeuristicSelection slot2_share=80 \\
        slot3_strategy=AdaptiveReset

The example above means:

    - slot 1 gets 60% of the global time/step budget
    - slot 2 gets 80% of what remains
    - slot 3 gets the rest

Suggested ParamILS params file:

    slot1_strategy {FirstAction,LeanCopCut,HeuristicSelection,SubgoalReordering,EntropyCut,AdaptiveReset,SmartCut} [LeanCopCut]
    slot1_share {5,10,15,20,25,30,35,40,50,60,70,80,90} [60]
    slot2_strategy {Off,FirstAction,LeanCopCut,HeuristicSelection,SubgoalReordering,EntropyCut,AdaptiveReset,SmartCut} [HeuristicSelection]
    slot2_share {5,10,15,20,25,30,35,40,50,60,70,80,90,100} [80]
    slot3_strategy {Off,FirstAction,LeanCopCut,HeuristicSelection,SubgoalReordering,EntropyCut,AdaptiveReset,SmartCut} [AdaptiveReset]

Suggested ParamILS scenario snippet:

    algo = python /abs/path/to/paramils_schedule_wrapper.py
    execdir = /abs/path/to/MyPyCop
    deterministic = 1
    run_obj = runtime
    overall_obj = mean1000
    cutoff_time = 60
    cutoff_length = 100000
    paramfile = /abs/path/to/params.txt
    instance_file = /abs/path/to/train_instances.txt
    outdir = /abs/path/to/paramils-output

Notes:

    - The wrapper treats any successful proof as "SAT" in ParamILS' output
      vocabulary. ParamILS mainly cares about success vs TIMEOUT for runtime
      tuning.
    - For deterministic schedules, use seed = -1 in the instance(-seed) file.
    - This script optimizes a static schedule, not an instance-dependent one.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

from connections.env import ConnectionEnv, Settings
from strategies import Strategy

SUCCESS_STATUSES = {"Theorem", "Theorem (SAT)"}
OFF_VALUES = {"off", "none", "-", ""}
MAX_SLOTS = 3
MP_CONTEXT = mp.get_context("fork")

@dataclass(frozen=True)
class Slot:
    index: int
    strategy_name: str
    time_budget: float
    step_budget: int | None

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ParamILS wrapper for static MyPyCop strategy schedules."
    )
    parser.add_argument("instance", help="Problem instance path or TPTP problem name.")
    parser.add_argument("run_obj", help="ParamILS run objective, e.g. runtime.")
    parser.add_argument("cutoff_time", help="Global wallclock cutoff for the scheduled run.")
    parser.add_argument(
        "cutoff_length",
        help="Global runlength cutoff, used here as a maximum step budget.",
    )
    parser.add_argument("seed", help="Run seed supplied by ParamILS.")
    parser.add_argument(
        "params",
        nargs=argparse.REMAINDER,
        help="Trailing ParamILS parameter assignments, e.g. slot1_strategy=LeanCopCut.",
    )
    return parser

def _safe_float(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}: {raw!r}")
    return value

def _safe_int(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"invalid {label}: {raw!r}")
    return value

def _parse_optional_length(raw: str) -> int | None:
    if raw in {"", "None", "none", "-1"}:
        return None
    value = _safe_int(raw, "cutoff_length")
    return None if value == 0 else value

def _parse_param_tokens(tokens: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if "=" in token:
            key, value = token.split("=", 1)
            params[key.lstrip("-")] = value
            i += 1
            continue

        key = token.lstrip("-")
        if not key:
            raise ValueError(f"malformed parameter token: {token!r}")
        if i + 1 >= len(tokens):
            raise ValueError(f"missing value for parameter {token!r}")
        params[key] = tokens[i + 1]
        i += 2
    return params

def _is_off(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in OFF_VALUES

def _share(params: dict[str, str], slot_index: int, default: int) -> int:
    key = f"slot{slot_index}_share"
    raw = params.get(key, str(default))
    value = _safe_int(raw, key)
    if not 0 <= value <= 100:
        raise ValueError(f"{key} must be between 0 and 100")
    return value

def _active_strategy_names(params: dict[str, str]) -> list[tuple[int, str]]:
    available = Strategy.strategies()
    active: list[tuple[int, str]] = []
    for index in range(1, MAX_SLOTS + 1):
        name = params.get(f"slot{index}_strategy")
        if _is_off(name):
            continue
        assert name is not None
        if name not in available:
            choices = ", ".join(sorted(available))
            raise ValueError(f"unknown strategy {name!r}; available strategies: {choices}")
        active.append((index, name))
    if not active:
        raise ValueError("at least one schedule slot must be active")
    return active

def build_schedule(params: dict[str, str], total_time: float, total_steps: int | None) -> list[Slot]:
    active = _active_strategy_names(params)
    remaining_time = total_time
    remaining_steps = total_steps
    schedule: list[Slot] = []

    for position, (index, strategy_name) in enumerate(active):
        is_last = position == len(active) - 1
        if is_last:
            slot_time = remaining_time
            slot_steps = remaining_steps
        else:
            share = _share(params, index, default=50)
            slot_time = remaining_time * share / 100.0
            slot_steps = None
            if remaining_steps is not None:
                slot_steps = max(1, int(round(remaining_steps * share / 100.0)))

        if slot_time <= 0:
            continue
        if slot_steps == 0:
            continue

        schedule.append(
            Slot(
                index=index,
                strategy_name=strategy_name,
                time_budget=slot_time,
                step_budget=slot_steps,
            )
        )

        remaining_time = max(0.0, remaining_time - slot_time)
        if remaining_steps is not None and slot_steps is not None:
            remaining_steps = max(0, remaining_steps - slot_steps)

    if not schedule:
        raise ValueError("all schedule slots ended up with zero budget")
    return schedule

def resolve_problem_path(path: str) -> str:
    problem_path = Path(path)

    for candidate in _candidate_paths(problem_path):
        if candidate.exists():
            return str(candidate)

    tptp_root = os.getenv("TPTP")
    if tptp_root is None:
        raise FileNotFoundError(f"could not find problem {path!r} and TPTP is not set")

    matches = _find_tptp_matches(problem_path, Path(tptp_root))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f"could not find problem {path!r} under {tptp_root}")

    formatted = ", ".join(str(match) for match in matches[:5])
    if len(matches) > 5:
        formatted += ", ..."
    raise FileNotFoundError(f"problem {path!r} is ambiguous under {tptp_root}: {formatted}")

def _candidate_paths(path: Path) -> list[Path]:
    candidates = [path]
    if path.suffix:
        return candidates
    candidates.append(path.with_suffix(".p"))
    return candidates

def _find_tptp_matches(path: Path, tptp_root: Path) -> list[Path]:
    matches: list[Path] = []
    for candidate in _candidate_paths(path):
        rooted = (tptp_root / candidate).resolve()
        if rooted.exists():
            matches.append(rooted)

    if matches:
        return _unique_paths(matches)

    for candidate in _candidate_paths(path):
        matches.extend(found.resolve() for found in tptp_root.rglob(candidate.name))
    return _unique_paths(matches)

def _unique_paths(paths: list[Path]) -> list[Path]:
    return sorted(set(paths))

def _run_strategy_worker(
    conn: Connection,
    problem_path: str,
    strategy_name: str,
    logic: str,
    max_steps: int | None,
) -> None:
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                logging.disable(logging.CRITICAL)
                env = ConnectionEnv(problem_path, Settings(logic=logic))
                strategy = Strategy.strategies()[strategy_name](env)
                env.reset()

                done, info = env.state.initial_status()
                steps = 0
                hit_max_steps = False
                start = time.perf_counter()

                while not done:
                    if max_steps is not None and steps >= max_steps:
                        hit_max_steps = True
                        break
                    action = strategy.step()
                    try:
                        _, _, done, info = env.step(action)
                    except RecursionError:
                        if max_steps is not None:
                            steps = max_steps - 1
                    steps += 1

                elapsed = time.perf_counter() - start
                status_info = info.get("status") if isinstance(info, dict) else info
                solved = status_info in SUCCESS_STATUSES
                status = "solved"
                if not solved:
                    status = "max_steps" if hit_max_steps else "unsolved"

                conn.send(
                    {
                        "status": status,
                        "solved": solved,
                        "steps": steps,
                        "time": elapsed,
                        "info": str(status_info) if status_info is not None else "",
                    }
                )
    except Exception as exc:
        conn.send(
            {
                "status": "error",
                "solved": False,
                "steps": 0,
                "time": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()

def _run_slot(problem_path: str, slot: Slot, logic: str) -> dict:
    parent_conn, child_conn = MP_CONTEXT.Pipe(duplex=False)
    process = MP_CONTEXT.Process(
        target=_run_strategy_worker,
        args=(child_conn, problem_path, slot.strategy_name, logic, slot.step_budget),
    )
    process.start()
    child_conn.close()

    if parent_conn.poll(slot.time_budget):
        result = parent_conn.recv()
        process.join()
        return result

    if process.is_alive():
        process.terminate()
        process.join()
    return {
        "status": "timeout",
        "solved": False,
        "steps": slot.step_budget if slot.step_budget is not None else -1,
        "time": slot.time_budget,
        "info": "",
    }

def run_schedule(problem_path: str, schedule: list[Slot], logic: str) -> tuple[bool, float, int]:
    total_runtime = 0.0
    total_steps = 0

    for slot in schedule:
        result = _run_slot(problem_path, slot, logic)
        total_runtime += min(slot.time_budget, float(result.get("time", slot.time_budget)))
        total_steps += int(result.get("steps", 0))
        if result.get("solved", False):
            return True, total_runtime, total_steps

    return False, total_runtime, total_steps

def _format_result(solved: bool, runtime: float, runlength: int, seed: str) -> str:
    solved_label = "SAT" if solved else "TIMEOUT"
    return f"Result for ParamILS: {solved_label}, {runtime:.6f}, {runlength}, -1, {seed}"

def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv[1:])

    try:
        problem = args.instance
        run_obj = args.run_obj
        cutoff_time = _safe_float(args.cutoff_time, "cutoff_time")
        cutoff_length = _parse_optional_length(args.cutoff_length)
        seed = args.seed
        params = _parse_param_tokens(args.params)
    except ValueError as exc:
        parser.error(str(exc))

    # ParamILS passes run_obj even if the wrapper does not need it. Keep the
    # parameter for interface compatibility and future extensions.
    _ = run_obj

    logic = params.pop("logic", "classical")
    problem_path = resolve_problem_path(problem)
    schedule = build_schedule(params, total_time=cutoff_time, total_steps=cutoff_length)
    solved, runtime, runlength = run_schedule(problem_path, schedule, logic=logic)
    print(_format_result(solved, runtime, runlength, seed))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
