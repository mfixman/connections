from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import TypeVar

from connections.runs import select_problem_paths
from connections.syntax.logic import Domain, Logic


LOGICS: tuple[Logic, ...] = ("classical", "intuitionistic", "D", "T", "S4", "S5")
DOMAINS: tuple[Domain, ...] = ("constant", "cumulative", "varying")
T = TypeVar("T")


def add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logic",
        default="classical",
        choices=LOGICS,
        help="Input logic (default: classical)",
    )
    parser.add_argument(
        "--domain",
        default="constant",
        choices=DOMAINS,
        help="Domain semantics (default: constant)",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Directory used to resolve included source files (repeatable)",
    )
    parser.add_argument(
        "--tptp",
        metavar="ROOT",
        help="Explicit TPTP root used for corpus and basename lookup",
    )
    parser.add_argument(
        "--settings",
        "--setting",
        action="append",
        default=[],
        dest="settings",
        help="leanCoP setting token (repeatable), e.g. cut or comp(7)",
    )
    parser.add_argument(
        "--backtrack",
        choices=("step", "maximal"),
        default="step",
        help="Backtracking granularity (default: step)",
    )
    parser.add_argument(
        "--max-steps",
        type=_nonnegative_int,
        default=1_000_000,
        help="Maximum policy calls per problem (default: 1000000)",
    )
    parser.add_argument(
        "--timeout",
        type=_nonnegative_float,
        default=120.0,
        metavar="SEC",
        help="Wall-clock budget per problem in seconds (default: 120)",
    )


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Partition the ordered problem list into N modulo slices",
    )
    parser.add_argument(
        "--part",
        type=_nonnegative_int,
        default=None,
        metavar="M",
        help="Run modulo slice M (zero based)",
    )


def finalize_split_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    split = args.split
    part = args.part
    if split is None and part is None:
        split_raw = os.getenv("SLURM_ARRAY_TASK_COUNT")
        part_raw = os.getenv("SLURM_ARRAY_TASK_ID")
        if split_raw is not None and part_raw is not None:
            try:
                split = int(split_raw)
                part = int(part_raw)
            except ValueError:
                parser.error(
                    "SLURM_ARRAY_TASK_COUNT and SLURM_ARRAY_TASK_ID must be integers"
                )
        else:
            split, part = 1, 0
    elif split is None:
        split = 1
    elif part is None:
        part = 0
    if split <= 0:
        parser.error("--split must be positive")
    if part < 0 or part >= split:
        parser.error("--part must satisfy 0 <= PART < SPLIT")
    args.split = split
    args.part = part


def partition_items(items: Sequence[T], *, split: int, part: int) -> tuple[T, ...]:
    if split <= 0 or part < 0 or part >= split:
        raise ValueError("partition requires split > 0 and 0 <= part < split")
    return tuple(item for index, item in enumerate(items) if index % split == part)


def source_file_dirs(args: argparse.Namespace) -> tuple[Path, ...]:
    directories = [Path(path).resolve() for path in args.source_dir]
    tptp_root = tptp_root_from_args(args)
    if tptp_root is not None and tptp_root not in directories:
        directories.append(tptp_root)
    return tuple(directories)


def tptp_root_from_args(args: argparse.Namespace) -> Path | None:
    raw = args.tptp
    if raw is None:
        return None
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TPTP root is not a directory: {root}")
    return root


@contextmanager
def resolved_problem_inputs(
    inputs: Sequence[str],
    *,
    args: argparse.Namespace,
    allow_raw: bool,
    default_to_all_tptp: bool,
) -> Iterator[tuple[Path, ...]]:
    if allow_raw and len(inputs) == 1:
        suffix = _raw_problem_suffix(inputs[0])
        if suffix is not None and not Path(inputs[0]).exists():
            with tempfile.TemporaryDirectory(prefix="raw-pycop-") as directory:
                path = Path(directory) / f"problem{suffix}"
                path.write_text(inputs[0], encoding="utf-8")
                yield (path,)
            return

    roots = tuple(inputs)
    tptp_root = tptp_root_from_args(args)
    if not roots:
        if not default_to_all_tptp:
            raise ValueError("pass a problem file, directory, codename, or raw problem")
        if tptp_root is None:
            raise FileNotFoundError(
                "no input was given and no TPTP root is configured; pass --tptp ROOT"
            )
        roots = (str(tptp_root / "Problems"),)

    resolved_roots = tuple(_resolve_root(root, tptp_root=tptp_root) for root in roots)
    problems = select_problem_paths(resolved_roots, pattern="*.p", recursive=True)
    if not problems:
        raise FileNotFoundError("no .p problem files found")
    yield partition_items(problems, split=args.split, part=args.part)


def determine_worker_count(total_tasks: int, requested: int | None) -> int:
    if total_tasks <= 0:
        return 1
    if requested is not None:
        return max(1, min(requested, total_tasks))
    slurm_raw = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_raw is not None:
        try:
            available = int(slurm_raw)
        except ValueError:
            available = 1
    else:
        available = os.cpu_count() or 1
    return max(1, min(available, total_tasks))


def _resolve_root(raw: str, *, tptp_root: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    if tptp_root is None:
        raise FileNotFoundError(
            f"could not find {raw!r}; pass --tptp ROOT for codenames and basenames"
        )
    candidates = [tptp_root / raw, tptp_root / "Problems" / raw]
    name = path.name
    if len(name) >= 3:
        candidates.append(tptp_root / "Problems" / name[:3] / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = tuple((tptp_root / "Problems").rglob(name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"could not find TPTP problem {raw!r} under {tptp_root}")
    raise FileNotFoundError(f"TPTP problem {raw!r} is ambiguous under {tptp_root}")


def _raw_problem_suffix(text: str) -> str | None:
    normalized = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    ).lower()
    if "qmf(" in normalized:
        return ".p"
    if "fof(" in normalized or "cnf(" in normalized:
        return ".p"
    return None


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


__all__ = [
    "DOMAINS",
    "LOGICS",
    "add_problem_arguments",
    "add_split_arguments",
    "determine_worker_count",
    "finalize_split_arguments",
    "partition_items",
    "resolved_problem_inputs",
    "source_file_dirs",
    "tptp_root_from_args",
]
