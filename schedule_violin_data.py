"""Build Mangla-style violin-plot data for time-based schedules.

The Mangla thesis Figure 2.6 compares strategies using the score from
Definition 2.4:

    r = (T - t) / (2T)
    score = 0.5 + r  on success
    score = r        otherwise

where ``T`` is the allocated time budget and ``t`` is the time consumed.

This script emulates each JSON schedule in ``schedules/`` against long-form
per-strategy results CSVs and writes one row per (schedule, problem), which is
the input needed to draw analogous violin plots for schedules.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

@dataclass(frozen=True)
class StrategyResult:
    status: str
    solved: bool
    time_ms: int | None

@dataclass(frozen=True)
class ScheduleSlot:
    strategy: str
    time_budget_ms: int

@dataclass(frozen=True)
class ScheduleSpec:
    path: Path
    name: str
    objective: str
    time_budget_ms: int
    slots: tuple[ScheduleSlot, ...]

@dataclass(frozen=True)
class EmulatedScheduleResult:
    problem: str
    schedule: str
    objective: str
    schedule_budget_ms: int
    solved: bool
    terminal_status: str
    time_spent_ms: int
    score: float
    solving_strategy: str
    solving_slot: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Mangla-style violin-plot data for schedules.",
    )
    parser.add_argument(
        "results_csv",
        nargs="+",
        help="One or more long-form results CSVs.",
    )
    parser.add_argument(
        "--schedule-glob",
        default="schedules/portfolio_*.json",
        help="Glob for time-based schedule JSON files.",
    )
    parser.add_argument(
        "--output",
        default="results/schedule_violin_data.csv",
        help="Long-form output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default="results/schedule_violin_summary.csv",
        help="Summary output CSV path.",
    )
    return parser.parse_args()

def _parse_int(text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None

def load_results(csv_paths: Iterable[str]) -> tuple[tuple[str, ...], dict[str, dict[str, StrategyResult]]]:
    problems: list[str] = []
    seen_problems: set[str] = set()
    by_problem: dict[str, dict[str, StrategyResult]] = {}

    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"problem", "strategy", "status", "solved", "time_ms"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing = required.difference(reader.fieldnames or ())
                raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

            for row in reader:
                problem = row["problem"].strip()
                strategy = row["strategy"].strip()
                if not problem or not strategy:
                    continue

                if problem not in seen_problems:
                    seen_problems.add(problem)
                    problems.append(problem)

                problem_results = by_problem.setdefault(problem, {})
                candidate = StrategyResult(
                    status=row["status"].strip(),
                    solved=row["solved"].strip().lower() == "true",
                    time_ms=_parse_int(row["time_ms"]),
                )

                current = problem_results.get(strategy)
                if current is None:
                    problem_results[strategy] = candidate
                    continue

                # Prefer successful runs, then the one with the smaller measured runtime,
                # then keep the earlier record for stability.
                current_key = (
                    1 if current.solved else 0,
                    -(current.time_ms if current.time_ms is not None else 10**18),
                )
                candidate_key = (
                    1 if candidate.solved else 0,
                    -(candidate.time_ms if candidate.time_ms is not None else 10**18),
                )
                if candidate_key > current_key:
                    problem_results[strategy] = candidate

    return tuple(problems), by_problem

def load_schedules(schedule_glob: str) -> tuple[ScheduleSpec, ...]:
    schedules: list[ScheduleSpec] = []
    for path in sorted(Path().glob(schedule_glob)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        time_budget_ms = payload.get("time_budget_ms")
        slots_payload = payload.get("slots")
        if not isinstance(time_budget_ms, int) or time_budget_ms <= 0:
            raise ValueError(f"{path} must define a positive integer time_budget_ms")
        if not isinstance(slots_payload, list) or not slots_payload:
            raise ValueError(f"{path} must define a non-empty slots list")

        slots: list[ScheduleSlot] = []
        total_slot_budget = 0
        for index, slot_payload in enumerate(slots_payload, start=1):
            if not isinstance(slot_payload, dict):
                raise ValueError(f"{path} slot {index} must be an object")
            strategy = slot_payload.get("strategy")
            slot_budget = slot_payload.get("time_budget_ms")
            if not isinstance(strategy, str) or not strategy:
                raise ValueError(f"{path} slot {index} must define a strategy")
            if not isinstance(slot_budget, int) or slot_budget <= 0:
                raise ValueError(f"{path} slot {index} must define a positive time_budget_ms")
            total_slot_budget += slot_budget
            slots.append(ScheduleSlot(strategy=strategy, time_budget_ms=slot_budget))

        if total_slot_budget > time_budget_ms:
            raise ValueError(
                f"{path} slot budgets sum to {total_slot_budget}ms, exceeding "
                f"schedule budget {time_budget_ms}ms"
            )

        schedules.append(
            ScheduleSpec(
                path=path,
                name=payload.get("name", path.stem),
                objective=payload.get("objective", ""),
                time_budget_ms=time_budget_ms,
                slots=tuple(slots),
            )
        )

    if not schedules:
        raise ValueError(f"no schedules matched {schedule_glob!r}")
    return tuple(schedules)

def mangla_score(success: bool, time_budget_ms: int, time_spent_ms: int) -> float:
    remainder = (time_budget_ms - time_spent_ms) / (2.0 * time_budget_ms)
    return 0.5 + remainder if success else remainder

def slot_time_spent(result: StrategyResult, allocation_ms: int) -> int:
    if result.solved:
        if result.time_ms is not None and result.time_ms <= allocation_ms:
            return result.time_ms
        return allocation_ms

    if result.time_ms is None:
        # Timeout/error rows without a measured runtime are conservatively treated as
        # consuming the full slot.
        return allocation_ms

    return min(result.time_ms, allocation_ms)

def emulate_schedule(
    problem: str,
    schedule: ScheduleSpec,
    problem_results: dict[str, StrategyResult],
) -> EmulatedScheduleResult:
    spent_ms = 0
    terminal_status = "not_solved"
    solving_strategy = ""
    solving_slot = 0

    for slot_index, slot in enumerate(schedule.slots, start=1):
        result = problem_results.get(slot.strategy)
        if result is None:
            raise KeyError(
                f"missing result for problem {problem!r} and strategy {slot.strategy!r}"
            )

        if result.solved and result.time_ms is not None and result.time_ms <= slot.time_budget_ms:
            spent_ms += result.time_ms
            terminal_status = "solved"
            solving_strategy = slot.strategy
            solving_slot = slot_index
            return EmulatedScheduleResult(
                problem=problem,
                schedule=schedule.name,
                objective=schedule.objective,
                schedule_budget_ms=schedule.time_budget_ms,
                solved=True,
                terminal_status=terminal_status,
                time_spent_ms=spent_ms,
                score=mangla_score(True, schedule.time_budget_ms, spent_ms),
                solving_strategy=solving_strategy,
                solving_slot=solving_slot,
            )

        spent_ms += slot_time_spent(result, slot.time_budget_ms)
        terminal_status = result.status

    spent_ms = min(spent_ms, schedule.time_budget_ms)
    return EmulatedScheduleResult(
        problem=problem,
        schedule=schedule.name,
        objective=schedule.objective,
        schedule_budget_ms=schedule.time_budget_ms,
        solved=False,
        terminal_status=terminal_status,
        time_spent_ms=spent_ms,
        score=mangla_score(False, schedule.time_budget_ms, spent_ms),
        solving_strategy="",
        solving_slot=0,
    )

def write_long_form(output_path: Path, rows: Iterable[EmulatedScheduleResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "problem",
                "schedule",
                "objective",
                "schedule_budget_ms",
                "solved",
                "terminal_status",
                "time_spent_ms",
                "time_spent_s",
                "score",
                "solving_strategy",
                "solving_slot",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.problem,
                    row.schedule,
                    row.objective,
                    row.schedule_budget_ms,
                    row.solved,
                    row.terminal_status,
                    row.time_spent_ms,
                    f"{row.time_spent_ms / 1000.0:.3f}",
                    f"{row.score:.12f}",
                    row.solving_strategy,
                    row.solving_slot,
                ]
            )

def write_summary(output_path: Path, rows: Iterable[EmulatedScheduleResult]) -> None:
    grouped: dict[str, list[EmulatedScheduleResult]] = {}
    for row in rows:
        grouped.setdefault(row.schedule, []).append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schedule",
                "objective",
                "schedule_budget_ms",
                "problems",
                "solved",
                "mean_score",
                "median_score",
                "mean_time_spent_s",
                "median_time_spent_s",
                "mean_time_spent_s_on_solved",
            ]
        )
        for schedule_name in sorted(grouped):
            schedule_rows = grouped[schedule_name]
            scores = [row.score for row in schedule_rows]
            spent_s = [row.time_spent_ms / 1000.0 for row in schedule_rows]
            solved_spent_s = [row.time_spent_ms / 1000.0 for row in schedule_rows if row.solved]
            first = schedule_rows[0]
            writer.writerow(
                [
                    schedule_name,
                    first.objective,
                    first.schedule_budget_ms,
                    len(schedule_rows),
                    sum(1 for row in schedule_rows if row.solved),
                    f"{sum(scores) / len(scores):.12f}",
                    f"{median(scores):.12f}",
                    f"{sum(spent_s) / len(spent_s):.6f}",
                    f"{median(spent_s):.6f}",
                    (
                        f"{sum(solved_spent_s) / len(solved_spent_s):.6f}"
                        if solved_spent_s else
                        ""
                    ),
                ]
            )

def main() -> int:
    args = parse_args()
    problems, by_problem = load_results(args.results_csv)
    schedules = load_schedules(args.schedule_glob)

    rows: list[EmulatedScheduleResult] = []
    for schedule in schedules:
        for problem in problems:
            rows.append(emulate_schedule(problem, schedule, by_problem[problem]))

    write_long_form(Path(args.output), rows)
    write_summary(Path(args.summary_output), rows)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
