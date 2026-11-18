# Finds TPTP problems on disk and loads them as matrices with their axiom and conjecture clauses.

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from connections.clausification import matrix_from_file
from connections.syntax.matrix import Matrix

tptp_category = re.compile(r"[A-Z]{3}")

def declared_tptp_status(path):
    status = re.compile(r"^\s*%\s*Status\s*:\s*(\S+)", re.IGNORECASE)

    with Path(path).open(encoding = "latin-1") as stream:
        for line in stream:
            match = status.match(line)
            if match:
                return match.group(1)

    return None

class NoProblemFilesError(FileNotFoundError):
    # A requested problem directory contains no immediate .p files.
    pass

@dataclass(frozen = True, slots = True)
class TPTPProblem:
    requested_path: str
    path: Path
    source_root: Path | None
    matrix: Matrix
    axiom_clause_ids: list[int]
    conjecture_clause_ids: list[int]

def find_tptp_root(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

        raise FileNotFoundError(f"TPTP root is not a directory: {candidate}")

    if os.environ.get("TPTP"):
        return find_tptp_root(os.environ["TPTP"])

    for candidate in default_tptp_roots():
        if (candidate / "Problems").is_dir() and (candidate / "Axioms").is_dir():
            return candidate.resolve()

    return None

def default_tptp_roots() -> list[Path]:
    from connections.corpora import workspace_root

    root = workspace_root(Path(__file__).resolve().parent) or Path.cwd()
    candidates = [Path.cwd() / "TPTP", root / "TPTP", root.parent / "TPTP"]
    for base in dict.fromkeys((root, Path.cwd())):
        candidates.extend(
            sorted(
                (base / "corpora").glob("TPTP*"),
                key = lambda path: list(map(int, re.findall(r"\d+", path.name))),
                reverse = True,
            )
        )

    candidates.append(Path.home() / "TPTP")
    return list(dict.fromkeys(candidates))

# External provers handle equality themselves, so the selector leaves the explicit equality axioms out unless
# --equality-axioms is given; the choice travels in the environment so that worker processes see it too.
def selector_equality_axioms():
    return os.environ.get("axiom_selector_equality_axioms") == "1"

def load_tptp_problem(
    problem: str | Path,
    *,
    tptp_root: str | Path | None = None,
    equality_axioms: bool | None = None,
) -> TPTPProblem:
    if equality_axioms is None:
        equality_axioms = selector_equality_axioms()

    requested = str(problem)
    path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)

    source_dirs = () if root is None else (root,)
    matrix = matrix_from_file(
        path,
        mark_conjecture = True,
        source_file_dirs = source_dirs,
        equality_axioms = equality_axioms,
    )

    conjectures = list(matrix.conjecture_clauses)
    conjecture_set = frozenset(conjectures)
    axioms = [index for index in range(len(matrix)) if index not in conjecture_set]
    return TPTPProblem(requested, path, root, matrix, axioms, conjectures)

def resolve_tptp_problem(
    problem: str | Path,
    *,
    tptp_root: str | Path | None = None,
) -> tuple[Path, Path | None]:
    root = find_tptp_root(tptp_root)
    resolved = expand_problem_inputs((str(problem),), tptp_root = root)
    if len(resolved) != 1:
        raise ValueError(
            f"expected one TPTP problem, but {problem!r} resolved to {len(resolved)} problems"
        )

    return Path(resolved[0]), root

def expand_problem_inputs(
    inputs: tuple[str, ...] | list[str],
    *,
    tptp_root: str | Path | None = None,
) -> list[str]:
    # Resolve files, flat directories, TPTP filenames, and category codes.

    root = find_tptp_root(tptp_root)
    problems: list[str] = []
    for raw in inputs:
        direct = Path(raw).expanduser()
        if direct.is_file():
            problems.append(str(direct.resolve()))
            continue

        if direct.is_dir():
            problems.extend(problems_in_directory(direct, requested = raw))
            continue

        if root is not None and (root / direct).is_file():
            problems.append(str((root / direct).resolve()))
            continue

        directory = tptp_input_directory(raw, root = root)
        if directory is not None:
            problems.extend(problems_in_directory(directory, requested = raw))
            continue

        if direct.name == raw and direct.suffix == ".p":
            problems.append(str(find_tptp_problem_by_name(raw, root = root)))
            continue

        raise FileNotFoundError(f"TPTP problem or directory not found: {raw}")

    return list(dict.fromkeys(problems))

def problems_in_directory(directory: Path, *, requested: str) -> list[str]:
    problems = [str(path.resolve()) for path in sorted(directory.glob("*.p")) if path.is_file()]

    if not problems:
        raise NoProblemFilesError(f"no .p problem files found in {requested!r}")

    return problems

def find_tptp_problem_by_name(filename: str, *, root: Path | None) -> Path:
    if root is None:
        raise FileNotFoundError(f"cannot find TPTP problem {filename!r} without a TPTP root")

    problems_root = root / "Problems"
    category = filename[:3]
    direct = problems_root / category / filename
    if tptp_category.fullmatch(category) and direct.is_file():
        return direct.resolve()

    matches = [
        path.resolve()
        for path in sorted(problems_root.glob(f"*/{filename}"))
        if path.is_file()
    ]

    if not matches:
        raise FileNotFoundError(f"TPTP problem {filename!r} not found under {problems_root}")

    if len(matches) > 1:
        raise ValueError(
            f"TPTP problem filename {filename!r} is ambiguous: "
            + ", ".join(map(str, matches))
        )

    return matches[0]

def problem_file(problem: str | Path, tptp_root: str | Path | None = None) -> str:
    # Datasets store bare file names; output needs the full path to read the declared status.
    if Path(problem).name != str(problem):
        return str(problem)

    try:
        return str(find_tptp_problem_by_name(str(problem), root = find_tptp_root(tptp_root)))
    except (OSError, ValueError):
        return str(problem)

def tptp_input_directory(raw: str, *, root: Path | None) -> Path | None:
    if root is None:
        return None

    relative = Path(raw)
    root_relative = root / relative
    if root_relative.is_dir():
        return root_relative.resolve()

    if tptp_category.fullmatch(raw):
        category = root / "Problems" / raw
        if category.is_dir():
            return category.resolve()

        raise FileNotFoundError(f"TPTP category directory not found: {category}")

    return None

def tptp_problems(*, tptp_root: str | Path | None = None) -> list[str]:
    root = find_tptp_root(tptp_root)
    if root is None:
        raise FileNotFoundError("no TPTP root found; pass PROBLEM paths, --tptp or set $TPTP")

    problems_root = root / "Problems"
    if not problems_root.is_dir():
        raise FileNotFoundError(f"TPTP Problems directory not found: {problems_root}")

    problems = [
        str(path.resolve())
        for path in sorted(problems_root.rglob("*.p"))
        if path.is_file() and ("-" in path.name or "+" in path.name)
    ]

    if not problems:
        raise RuntimeError(f"no .p problem files found under {problems_root}")

    return problems

def declared_non_refutable(path):
    outcomes = {
        "satisfiable": "DeclaredSatisfiable",
        "countersatisfiable": "DeclaredCounterSatisfiable",
    }

    status = declared_tptp_status(path)
    return None if status is None else outcomes.get(status.casefold())
