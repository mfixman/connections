"""Optimize a time-based static strategy allocation from long-form results CSV.

The expected input format matches ``compare_strategies.py`` output:

    problem,strategy,status,solved,steps,time_ms,info,error
    AGT001+1.p,FirstAction,solved,True,4,1,,
    AGT001+1.p,LeanCopCut,solved,True,4,1,,
    ...

The optimizer chooses per-strategy time allocations ``x_s`` in whole seconds such that:

* ``sum_s x_s <= budget_seconds``;
* a problem is solved iff at least one strategy gets enough allocated time to solve it;
* the objective is lexicographic:
  1. maximize the total number of solved problems;
  2. among ties, maximize the number of solved problems that ``baseline`` does not solve.

Because the objective only depends on whether a problem is solved within the global budget,
the order of strategy slots does not affect the score; only the time allocation vector matters.
"""

from __future__ import annotations

import argparse
import csv
import math
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

DEFAULT_BUDGET_SECONDS = 300
DEFAULT_BASELINE = "LeanCopCut"
DEFAULT_EXCLUDE = ("InitialSchedule",)

@dataclass(frozen=True)
class SolveRecord:
    problem: str
    strategy: str
    seconds_required: int

@dataclass(frozen=True)
class ProblemScore:
    solved: int
    unique_vs_baseline: int

@dataclass(frozen=True)
class AllocationResult:
    objective_value: float
    problem_score: ProblemScore
    allocations: dict[str, int]

@dataclass(frozen=True)
class OrderedScheduleResult:
    order: tuple[str, ...]
    solved: int
    total_solved_seconds: float

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a static time allocation across strategies from long-form results CSV.",
    )
    parser.add_argument(
        "results_csv",
        help="Path to a CSV like results/results_1m_300.csv.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=DEFAULT_BUDGET_SECONDS,
        help=f"Total time budget in seconds (default: {DEFAULT_BUDGET_SECONDS}).",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help=f"Control strategy used for the secondary objective (default: {DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Comma-separated strategies to keep. Default: all strategies except excluded ones.",
    )
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help=(
            "Comma-separated strategies to drop. "
            f"Default: {', '.join(DEFAULT_EXCLUDE)}."
        ),
    )
    parser.add_argument(
        "--min-seconds",
        type=int,
        default=1,
        help=(
            "Minimum whole-second allocation needed for any successful run. "
            "Use 1 to avoid 0-second slots caused by rounded time_ms values (default: 1)."
        ),
    )
    return parser

def _parse_name_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())

def _seconds_required(time_ms_text: str, min_seconds: int) -> int | None:
    text = time_ms_text.strip()
    if not text:
        return None
    try:
        time_ms = int(text)
    except ValueError:
        return None
    return max(min_seconds, math.ceil(time_ms / 1000.0))

def load_solve_records(
    csv_path: str,
    budget_seconds: int,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    min_seconds: int = 1,
) -> tuple[tuple[str, ...], tuple[str, ...], list[SolveRecord], dict[str, dict[str, int]]]:
    include_set = set(include)
    exclude_set = set(exclude)

    selected_strategies: list[str] = []
    selected_strategy_seen: set[str] = set()
    problems: list[str] = []
    problem_seen: set[str] = set()

    # Keep the fastest successful run per (problem, strategy).
    solve_seconds: dict[str, dict[str, int]] = defaultdict(dict)

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"problem", "strategy", "status", "time_ms"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            missing = required_fields.difference(reader.fieldnames or ())
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            problem = row["problem"].strip()
            strategy = row["strategy"].strip()
            status = row["status"].strip()

            if not problem or not strategy:
                continue

            if problem not in problem_seen:
                problem_seen.add(problem)
                problems.append(problem)

            selected = (not include_set or strategy in include_set) and strategy not in exclude_set
            if selected and strategy not in selected_strategy_seen:
                selected_strategy_seen.add(strategy)
                selected_strategies.append(strategy)

            if status != "solved":
                continue

            seconds = _seconds_required(row["time_ms"], min_seconds=min_seconds)
            if seconds is None or seconds > budget_seconds:
                continue

            current = solve_seconds[problem].get(strategy)
            if current is None or seconds < current:
                solve_seconds[problem][strategy] = seconds

    strategies = tuple(selected_strategies)
    ordered_problems = tuple(problems)
    records = [
        SolveRecord(problem=problem, strategy=strategy, seconds_required=seconds)
        for problem in ordered_problems
        for strategy, seconds in solve_seconds.get(problem, {}).items()
        if strategy in selected_strategy_seen
    ]
    return strategies, ordered_problems, records, solve_seconds

def optimize_allocations(
    strategies: tuple[str, ...],
    problems: tuple[str, ...],
    solve_records: list[SolveRecord],
    solve_seconds: dict[str, dict[str, int]],
    budget_seconds: int,
    baseline: str,
) -> AllocationResult:
    if not strategies:
        raise ValueError("no strategies remain after include/exclude filtering")
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    baseline_in_data = baseline in strategies or any(
        baseline in by_strategy for by_strategy in solve_seconds.values()
    )
    if not baseline_in_data:
        raise ValueError(f"baseline {baseline!r} is not present in the results data")

    strategy_count = len(strategies)
    problem_count = len(problems)

    problem_index = {name: idx for idx, name in enumerate(problems)}

    # The score only changes when a strategy allocation crosses an integer second threshold
    # required by at least one solved run. Keep 0 as an explicit "unused" option.
    thresholds_by_strategy: dict[str, tuple[int, ...]] = {}
    for strategy in strategies:
        thresholds = {0}
        for per_problem in solve_seconds.values():
            seconds = per_problem.get(strategy)
            if seconds is not None and seconds <= budget_seconds:
                thresholds.add(seconds)
        thresholds_by_strategy[strategy] = tuple(sorted(thresholds))

    y_keys: list[tuple[str, int]] = []
    y_index: dict[tuple[str, int], int] = {}
    for strategy in strategies:
        for threshold in thresholds_by_strategy[strategy]:
            y_index[(strategy, threshold)] = len(y_keys)
            y_keys.append((strategy, threshold))

    y_count = len(y_keys)
    z_offset = y_count
    variable_count = y_count + problem_count

    integrality = np.ones(variable_count, dtype=int)
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    row_counter = 0

    # Budget: sum_{s,t} t * y_{s,t} <= budget.
    for (strategy, threshold), var_idx in y_index.items():
        rows.append(row_counter)
        cols.append(var_idx)
        values.append(float(threshold))
    lb.append(0.0)
    ub.append(float(budget_seconds))
    row_counter += 1

    # Choose exactly one threshold per strategy.
    for strategy in strategies:
        for threshold in thresholds_by_strategy[strategy]:
            rows.append(row_counter)
            cols.append(y_index[(strategy, threshold)])
            values.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        row_counter += 1

    # z_p <= sum selected thresholds that can solve p. Because exactly one threshold is chosen
    # per strategy, this correctly models the OR over strategies.
    by_problem_strategy: dict[str, dict[str, int]] = defaultdict(dict)
    for record in solve_records:
        current = by_problem_strategy[record.problem].get(record.strategy)
        if current is None or record.seconds_required < current:
            by_problem_strategy[record.problem][record.strategy] = record.seconds_required

    for problem in problems:
        problem_idx = problem_index[problem]
        rows.append(row_counter)
        cols.append(z_offset + problem_idx)
        values.append(1.0)
        for strategy, required in by_problem_strategy.get(problem, {}).items():
            for threshold in thresholds_by_strategy[strategy]:
                if threshold >= required:
                    rows.append(row_counter)
                    cols.append(y_index[(strategy, threshold)])
                    values.append(-1.0)
        lb.append(-np.inf)
        ub.append(0.0)
        row_counter += 1

    constraints = LinearConstraint(
        coo_array((values, (rows, cols)), shape=(row_counter, variable_count)).tocsr(),
        np.array(lb, dtype=float),
        np.array(ub, dtype=float),
    )

    # Maximize M * solved + unique_vs_baseline, then minimize total allocated time slightly.
    # SciPy MILP minimizes, so negate the rewards.
    objective = np.zeros(variable_count, dtype=float)
    lexicographic_weight = problem_count + 1
    for problem_idx, problem in enumerate(problems):
        baseline_seconds = solve_seconds.get(problem, {}).get(baseline)
        bonus = 1 if baseline_seconds is None or baseline_seconds > budget_seconds else 0
        objective[z_offset + problem_idx] = -(lexicographic_weight + bonus)

    # Tertiary tie-breaker: prefer smaller total allocations among equal coverage solutions.
    for (strategy, threshold), var_idx in y_index.items():
        objective[var_idx] = 1e-3 * threshold

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")

    solution = result.x
    allocations: dict[str, int] = {}
    for strategy in strategies:
        chosen_threshold = 0
        for threshold in thresholds_by_strategy[strategy]:
            if round(solution[y_index[(strategy, threshold)]]) >= 1:
                chosen_threshold = threshold
                break
        allocations[strategy] = chosen_threshold

    solved = 0
    unique_vs_baseline = 0
    for problem_idx, problem in enumerate(problems):
        if round(solution[z_offset + problem_idx]) >= 1:
            solved += 1
            baseline_seconds = solve_seconds.get(problem, {}).get(baseline)
            if baseline_seconds is None or baseline_seconds > budget_seconds:
                unique_vs_baseline += 1

    return AllocationResult(
        objective_value=float(result.fun),
        problem_score=ProblemScore(solved=solved, unique_vs_baseline=unique_vs_baseline),
        allocations=allocations,
    )

def _evaluate_order(
    order: tuple[str, ...],
    allocations: dict[str, int],
    solve_seconds: dict[str, dict[str, int]],
) -> OrderedScheduleResult:
    solved = 0
    total_solved_seconds = 0.0

    for problem, by_strategy in solve_seconds.items():
        spent_before = 0.0
        solved_here = False
        for strategy in order:
            allocation = allocations[strategy]
            required = by_strategy.get(strategy)
            if required is not None and required <= allocation:
                total_solved_seconds += spent_before + required
                solved += 1
                solved_here = True
                break
            spent_before += allocation
        if not solved_here:
            continue

    return OrderedScheduleResult(
        order=order,
        solved=solved,
        total_solved_seconds=total_solved_seconds,
    )

def best_order_for_allocations(
    allocations: dict[str, int],
    solve_seconds: dict[str, dict[str, int]],
) -> OrderedScheduleResult:
    active = tuple(
        strategy for strategy, seconds in allocations.items()
        if seconds > 0
    )
    if not active:
        return OrderedScheduleResult(order=(), solved=0, total_solved_seconds=0.0)

    best: OrderedScheduleResult | None = None
    for order in itertools.permutations(active):
        candidate = _evaluate_order(order, allocations, solve_seconds)
        if best is None:
            best = candidate
            continue

        lhs = (
            candidate.solved,
            -candidate.total_solved_seconds,
            tuple((-allocations[s], s) for s in candidate.order),
        )
        rhs = (
            best.solved,
            -best.total_solved_seconds,
            tuple((-allocations[s], s) for s in best.order),
        )
        if lhs > rhs:
            best = candidate

    assert best is not None
    return best

def _format_order(order: tuple[str, ...], allocations: dict[str, int]) -> str:
    return " -> ".join(f"{strategy}@{allocations[strategy]}s" for strategy in order) or "(empty)"

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    include = _parse_name_list(args.include)
    exclude = _parse_name_list(args.exclude)

    strategies, problems, records, solve_seconds = load_solve_records(
        args.results_csv,
        budget_seconds=args.budget_seconds,
        include=include,
        exclude=exclude,
        min_seconds=args.min_seconds,
    )
    allocation = optimize_allocations(
        strategies=strategies,
        problems=problems,
        solve_records=records,
        solve_seconds=solve_seconds,
        budget_seconds=args.budget_seconds,
        baseline=args.baseline,
    )
    ordered = best_order_for_allocations(allocation.allocations, solve_seconds)

    print(f"Results CSV: {Path(args.results_csv)}")
    print(f"Problems: {len(problems)}")
    print(f"Strategies considered: {', '.join(strategies)}")
    print(f"Budget: {args.budget_seconds}s")
    print(f"Baseline: {args.baseline}")
    print()
    print("Optimal allocation:")
    for strategy, seconds in sorted(
        allocation.allocations.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if seconds <= 0:
            continue
        print(f"  {strategy}: {seconds}s")
    print(f"  total allocated: {sum(allocation.allocations.values())}s")
    print()
    print("Coverage:")
    print(f"  solved: {allocation.problem_score.solved}/{len(problems)}")
    print(
        "  solved but not by baseline: "
        f"{allocation.problem_score.unique_vs_baseline}/{len(problems)}"
    )
    print()
    print("Best slot order for earlier solved problems:")
    print(f"  {_format_order(ordered.order, allocation.allocations)}")
    if ordered.solved:
        mean_solved_seconds = ordered.total_solved_seconds / ordered.solved
        print(f"  mean time to solve (among solved problems): {mean_solved_seconds:.2f}s")
    else:
        print("  mean time to solve (among solved problems): n/a")
    print()
    print(
        "Note: the allocation optimizer ignores slot order for coverage, then the script chooses "
        "the best order for the same allocation by minimizing solved time."
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
