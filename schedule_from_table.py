"""Infer a static strategy schedule from a wide CSV result table.

The expected input format matches ``results/full_table.csv``:

    problem,StrategyA,StrategyB,...
    PROB001,12,timeout,7
    PROB002,max_steps,44,timeout

Each numeric cell is interpreted as the number of steps needed to solve the
problem with that strategy. Any non-numeric cell is treated as unsolved.

The optimizer searches schedules in the same style as
``paramils_schedule_wrapper.py``:

* a schedule has up to ``max_slots`` active slots;
* every non-final slot chooses a strategy and a percentage share of the
  remaining step budget;
* the final slot gets the rest of the global step budget.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SHARES = (5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90)
DEFAULT_PENALTY_MULTIPLIER = 10
DEFAULT_MAX_SLOTS = 3

@dataclass(frozen=True)
class SlotSpec:
    strategy: str
    share: int | None
    cap: int

@dataclass(frozen=True)
class ScheduleScore:
    solved: int
    penalized_steps: int
    solved_steps: int

@dataclass(frozen=True)
class SearchResult:
    schedule: tuple[SlotSpec, ...]
    score: ScheduleScore
    problems: int

@dataclass(frozen=True)
class TableData:
    strategies: tuple[str, ...]
    problems: tuple[str, ...]
    steps_by_strategy: dict[str, tuple[int | None, ...]]

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer a static MyPyCop schedule from a wide CSV result table.",
    )
    parser.add_argument(
        "table",
        help="Path to a CSV like results/full_table.csv.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Global step budget for the schedule.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=DEFAULT_MAX_SLOTS,
        help=f"Maximum number of active schedule slots (default: {DEFAULT_MAX_SLOTS}).",
    )
    parser.add_argument(
        "--shares",
        default=",".join(str(x) for x in DEFAULT_SHARES),
        help="Comma-separated non-final share percentages (default matches ParamILS wrapper).",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Comma-separated strategy names to keep. Default: all table columns except problem.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated strategy names to drop.",
    )
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Allow the same strategy to appear in multiple slots.",
    )
    parser.add_argument(
        "--penalty-multiplier",
        type=int,
        default=DEFAULT_PENALTY_MULTIPLIER,
        help=(
            "Penalty multiplier for unsolved problems when tie-breaking on total "
            f"steps (default: {DEFAULT_PENALTY_MULTIPLIER})."
        ),
    )
    return parser

def _parse_name_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())

def _parse_share_list(raw: str) -> tuple[int, ...]:
    shares = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid share value {token!r}") from exc
        if not 1 <= value <= 100:
            raise ValueError(f"share must be between 1 and 100: {value}")
        if value not in seen:
            shares.append(value)
            seen.add(value)
    if not shares:
        raise ValueError("at least one share value is required")
    return tuple(sorted(shares))

def _parse_step_cell(raw: str) -> int | None:
    text = raw.strip()
    if not text:
        return None
    normalized = text.replace(",", "").replace("_", "")
    if normalized.isdigit():
        return int(normalized)
    return None

def load_table(
    csv_path: str,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> TableData:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header row")
        if "problem" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'problem' column")

        all_strategies = [name for name in reader.fieldnames if name != "problem"]
        include_set = set(include)
        exclude_set = set(exclude)

        if include_set:
            strategies = [name for name in all_strategies if name in include_set]
            missing = sorted(include_set - set(all_strategies))
            if missing:
                raise ValueError(f"unknown included strategies: {', '.join(missing)}")
        else:
            strategies = list(all_strategies)

        strategies = [name for name in strategies if name not in exclude_set]
        if not strategies:
            raise ValueError("no strategies remain after include/exclude filtering")

        problems: list[str] = []
        steps: dict[str, list[int | None]] = {name: [] for name in strategies}

        for row in reader:
            problem = row.get("problem", "").strip()
            if not problem:
                continue
            problems.append(problem)
            for strategy in strategies:
                steps[strategy].append(_parse_step_cell(row.get(strategy, "")))

    return TableData(
        strategies=tuple(strategies),
        problems=tuple(problems),
        steps_by_strategy={name: tuple(values) for name, values in steps.items()},
    )

def build_schedule_caps(total_budget: int, shares: Sequence[int]) -> tuple[int, ...]:
    remaining = total_budget
    caps: list[int] = []
    for share in shares:
        cap = max(1, int(round(remaining * share / 100.0)))
        cap = min(cap, remaining)
        caps.append(cap)
        remaining -= cap
    caps.append(remaining)
    return tuple(cap for cap in caps if cap > 0)

def _compare_scores(lhs: ScheduleScore, rhs: ScheduleScore) -> int:
    lhs_key = (lhs.solved, -lhs.penalized_steps, -lhs.solved_steps)
    rhs_key = (rhs.solved, -rhs.penalized_steps, -rhs.solved_steps)
    if lhs_key < rhs_key:
        return -1
    if lhs_key > rhs_key:
        return 1
    return 0

def evaluate_schedule(
    table: TableData,
    schedule: Sequence[SlotSpec],
    total_budget: int,
    penalty_multiplier: int = DEFAULT_PENALTY_MULTIPLIER,
) -> ScheduleScore:
    solved = 0
    penalized_steps = 0
    solved_steps = 0

    for problem_index in range(len(table.problems)):
        spent = 0
        done = False
        for slot in schedule:
            required = table.steps_by_strategy[slot.strategy][problem_index]
            if required is not None and required <= slot.cap:
                spent += required
                solved += 1
                solved_steps += spent
                penalized_steps += spent
                done = True
                break
            spent += slot.cap
        if not done:
            penalized_steps += penalty_multiplier * total_budget

    return ScheduleScore(
        solved=solved,
        penalized_steps=penalized_steps,
        solved_steps=solved_steps,
    )

def _schedule_signature(schedule: Sequence[SlotSpec]) -> tuple[tuple[str, int | None, int], ...]:
    return tuple((slot.strategy, slot.share, slot.cap) for slot in schedule)

def _compare_schedules(
    lhs_schedule: Sequence[SlotSpec],
    lhs_score: ScheduleScore,
    rhs_schedule: Sequence[SlotSpec],
    rhs_score: ScheduleScore,
) -> int:
    cmp = _compare_scores(lhs_score, rhs_score)
    if cmp != 0:
        return cmp
    lhs_sig = _schedule_signature(lhs_schedule)
    rhs_sig = _schedule_signature(rhs_schedule)
    if lhs_sig < rhs_sig:
        return 1
    if lhs_sig > rhs_sig:
        return -1
    return 0

def _possible_caps(total_budget: int, max_slots: int, shares: Sequence[int]) -> tuple[int, ...]:
    caps = {total_budget}
    frontier = {total_budget}
    for _ in range(max_slots - 1):
        next_frontier: set[int] = set()
        for remaining in frontier:
            for share in shares:
                current = max(1, int(round(remaining * share / 100.0)))
                current = min(current, remaining)
                rest = remaining - current
                if current > 0:
                    caps.add(current)
                if rest > 0:
                    caps.add(rest)
                    next_frontier.add(rest)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted(caps))

def _build_solved_masks(table: TableData, caps: Sequence[int]) -> dict[str, dict[int, int]]:
    masks: dict[str, dict[int, int]] = {}
    for strategy in table.strategies:
        by_cap: dict[int, int] = {}
        steps = table.steps_by_strategy[strategy]
        for cap in caps:
            mask = 0
            for index, required in enumerate(steps):
                if required is not None and required <= cap:
                    mask |= 1 << index
            by_cap[cap] = mask
        masks[strategy] = by_cap
    return masks

def find_best_schedule(
    table: TableData,
    total_budget: int,
    max_slots: int = DEFAULT_MAX_SLOTS,
    shares: Sequence[int] = DEFAULT_SHARES,
    allow_repeat: bool = False,
    penalty_multiplier: int = DEFAULT_PENALTY_MULTIPLIER,
) -> SearchResult:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    if not table.strategies:
        raise ValueError("table must contain at least one strategy")

    caps = _possible_caps(total_budget, max_slots=max_slots, shares=shares)
    solved_masks = _build_solved_masks(table, caps)

    best_schedule: tuple[SlotSpec, ...] | None = None
    best_score: ScheduleScore | None = None

    def consider(schedule: tuple[SlotSpec, ...], coverage_mask: int) -> None:
        nonlocal best_schedule, best_score
        solved = coverage_mask.bit_count()
        if best_score is not None and solved < best_score.solved:
            return
        score = evaluate_schedule(
            table,
            schedule,
            total_budget=total_budget,
            penalty_multiplier=penalty_multiplier,
        )
        if best_schedule is None or best_score is None:
            best_schedule = schedule
            best_score = score
            return
        if _compare_schedules(schedule, score, best_schedule, best_score) > 0:
            best_schedule = schedule
            best_score = score

    def recurse(
        remaining_budget: int,
        slots_left: int,
        prefix: tuple[SlotSpec, ...],
        used_strategies: tuple[str, ...],
        coverage_mask: int,
    ) -> None:
        if slots_left <= 0 or remaining_budget <= 0:
            return

        available = table.strategies
        if not allow_repeat:
            used = set(used_strategies)
            available = tuple(name for name in table.strategies if name not in used)
        if not available:
            return

        is_final = slots_left == 1
        if is_final:
            for strategy in available:
                cap = remaining_budget
                slot = SlotSpec(strategy=strategy, share=None, cap=cap)
                coverage = coverage_mask | solved_masks[strategy][cap]
                consider(prefix + (slot,), coverage)
            return

        for strategy in available:
            for share in shares:
                cap = max(1, int(round(remaining_budget * share / 100.0)))
                cap = min(cap, remaining_budget)
                rest = remaining_budget - cap
                if rest <= 0:
                    continue
                slot = SlotSpec(strategy=strategy, share=share, cap=cap)
                new_prefix = prefix + (slot,)
                new_coverage = coverage_mask | solved_masks[strategy][cap]
                consider(new_prefix, new_coverage)
                recurse(
                    rest,
                    slots_left - 1,
                    new_prefix,
                    used_strategies + (strategy,),
                    new_coverage,
                )

    recurse(
        remaining_budget=total_budget,
        slots_left=max_slots,
        prefix=(),
        used_strategies=(),
        coverage_mask=0,
    )

    assert best_schedule is not None
    assert best_score is not None
    return SearchResult(
        schedule=best_schedule,
        score=best_score,
        problems=len(table.problems),
    )

def best_single_strategy(
    table: TableData,
    total_budget: int,
    penalty_multiplier: int = DEFAULT_PENALTY_MULTIPLIER,
) -> SearchResult:
    best: SearchResult | None = None
    for strategy in table.strategies:
        schedule = (SlotSpec(strategy=strategy, share=None, cap=total_budget),)
        score = evaluate_schedule(
            table,
            schedule,
            total_budget=total_budget,
            penalty_multiplier=penalty_multiplier,
        )
        candidate = SearchResult(schedule=schedule, score=score, problems=len(table.problems))
        if best is None or _compare_schedules(
            candidate.schedule,
            candidate.score,
            best.schedule,
            best.score,
        ) > 0:
            best = candidate
    assert best is not None
    return best

def _format_schedule(schedule: Sequence[SlotSpec]) -> str:
    parts = []
    for index, slot in enumerate(schedule, start=1):
        if slot.share is None:
            parts.append(f"{index}:{slot.strategy}@{slot.cap}")
        else:
            parts.append(f"{index}:{slot.strategy}@{slot.cap} (share {slot.share}% of remaining)")
    return " -> ".join(parts)

def _format_paramils(schedule: Sequence[SlotSpec]) -> str:
    tokens: list[str] = []
    for index, slot in enumerate(schedule, start=1):
        tokens.append(f"slot{index}_strategy={slot.strategy}")
        if slot.share is not None:
            tokens.append(f"slot{index}_share={slot.share}")
    return " ".join(tokens)

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    shares = _parse_share_list(args.shares)
    include = _parse_name_list(args.include)
    exclude = _parse_name_list(args.exclude)

    table = load_table(args.table, include=include, exclude=exclude)
    best = find_best_schedule(
        table,
        total_budget=args.budget,
        max_slots=args.max_slots,
        shares=shares,
        allow_repeat=args.allow_repeat,
        penalty_multiplier=args.penalty_multiplier,
    )
    single = best_single_strategy(
        table,
        total_budget=args.budget,
        penalty_multiplier=args.penalty_multiplier,
    )

    print(f"Table: {Path(args.table)}")
    print(f"Problems: {best.problems}")
    print(f"Strategies: {', '.join(table.strategies)}")
    print(f"Budget: {args.budget} steps")
    print()
    print("Best schedule:")
    print(f"  {_format_schedule(best.schedule)}")
    print(f"  solved: {best.score.solved}/{best.problems}")
    print(f"  penalized steps: {best.score.penalized_steps}")
    if best.score.solved:
        mean_solved = best.score.solved_steps / best.score.solved
        print(f"  mean solved steps: {mean_solved:.2f}")
    else:
        print("  mean solved steps: n/a")
    print(f"  ParamILS params: {_format_paramils(best.schedule)}")
    print()
    print("Best single strategy:")
    print(f"  {_format_schedule(single.schedule)}")
    print(f"  solved: {single.score.solved}/{single.problems}")
    print(f"  penalized steps: {single.score.penalized_steps}")
    if single.score.solved:
        mean_single = single.score.solved_steps / single.score.solved
        print(f"  mean solved steps: {mean_single:.2f}")
    else:
        print("  mean solved steps: n/a")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
