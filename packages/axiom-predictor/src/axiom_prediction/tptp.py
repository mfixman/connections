from __future__ import annotations

from .choices import ProverPolicy

from dataclasses import dataclass
import os
from pathlib import Path
import re

from connections.clausification import matrix_from_file
from connections.agent.sat import SATCoPCon, SATResetCoP
from connections.interaction.run import Problem, run_schedule
from connections.interaction.szs import SUCCESS as success
from connections.interaction.strategy import MatrixOptions, PolicyOptions, Strategy, StrategySchedule
from connections.syntax.matrix import Matrix
from connections.recursion import deep_recursion

from .data import AxiomTrainingExample, sat_core_clause_ids, training_example_from_sat_core

from .limits import default_collection_step_limit, default_collection_timeout_s, collection_budget

_tptp_category = re.compile(r"[A-Z]{3}")

def declared_tptp_status(path):
    _status = re.compile(r"^\s*%\s*Status\s*:\s*(\S+)", re.IGNORECASE)

    with Path(path).open(encoding = "latin-1") as stream:
        for line in stream:
            match = _status.match(line)
            if match:
                return match.group(1)

    return None

class NoProblemFilesError(FileNotFoundError):
    """A requested problem directory contains no immediate .p files."""

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
                key = lambda path: [int(n) for n in re.findall(r"\d+", path.name)],
                reverse = True,
            )
        )

    candidates.append(Path.home() / "TPTP")
    return list(dict.fromkeys(candidates))

def load_tptp_problem(
    problem: str | Path,
    *,
    tptp_root: str | Path | None = None,
) -> TPTPProblem:
    requested = str(problem)
    path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)

    source_dirs = () if root is None else (root,)
    matrix = matrix_from_file(
        path,
        mark_conjecture = True,
        source_file_dirs = source_dirs,
    )

    conjectures = list(matrix.conjecture_clauses)
    conjecture_set = frozenset(conjectures)
    axioms = [index for index in range(len(matrix)) if index not in conjecture_set]
    from .graph import validate_clause_ids

    validate_clause_ids(
        matrix,
        axiom_clause_ids = axioms,
        conjecture_clause_ids = conjectures,
    )

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
    """Resolve files, flat directories, TPTP filenames, and category codes."""

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
    if _tptp_category.fullmatch(category) and direct.is_file():
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
            + ", ".join(str(path) for path in matches)
        )

    return matches[0]

def problem_input_directory(
    problem: str | Path,
    *,
    tptp_root: str | Path | None = None,
) -> Path | None:
    """Return the resolved directory denoted by one CLI problem input."""

    root = find_tptp_root(tptp_root)
    direct = Path(problem).expanduser()
    if direct.is_dir():
        return direct.resolve()

    return tptp_input_directory(str(problem), root = root)

def tptp_input_directory(raw: str, *, root: Path | None) -> Path | None:
    if root is None:
        return None

    relative = Path(raw)
    root_relative = root / relative
    if root_relative.is_dir():
        return root_relative.resolve()

    if _tptp_category.fullmatch(raw):
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

@deep_recursion()
@collection_budget
def collect_proof_example(
    problem: str | Path,
    *,
    tptp_root: str | Path | None = None,
    step_limit: int = default_collection_step_limit,
    timeout_s: float = default_collection_timeout_s,
    sat_policy: str = ProverPolicy.SatResetCoP,
) -> tuple[AxiomTrainingExample | None, str]:
    _non_refutable_declared_outcomes = {
        "satisfiable": "DeclaredSatisfiable",
        "countersatisfiable": "DeclaredCounterSatisfiable",
    }

    policies = {ProverPolicy.SatCoP: SATCoPCon, ProverPolicy.SatResetCoP: SATResetCoP}
    sat_policy = ProverPolicy(sat_policy)
    if sat_policy not in policies:
        raise ValueError(f"unsupported SAT policy: {sat_policy!r}")

    path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
    declared_status = declared_tptp_status(path)
    declared_outcome = (
        None
        if declared_status is None
        else _non_refutable_declared_outcomes.get(declared_status.casefold())
    )

    if declared_outcome is not None:
        # Validate parsing before skipping so unsupported typed problems still
        # count as unparseable rather than as ordinary search failures.
        matrix_from_file(
            path,
            mark_conjecture = True,
            source_file_dirs = () if root is None else (root,),
        )

        return None, declared_outcome

    strategy = Strategy(
        matrix = MatrixOptions(mark_conjecture = True),
        policy = PolicyOptions(
            policy_class = policies[sat_policy],
            args = {"debug_sat_core": True},
        ),
    )

    agent = policies[sat_policy](debug_sat_core = True)
    result = run_schedule(
        Problem(
            path,
            logic = "classical",
            domain = "constant",
            source_file_dirs = () if root is None else (root,),
        ),
        agent = agent,
        schedule = StrategySchedule.single(
            strategy,
            steps = step_limit,
            timeout_seconds = timeout_s,
        ),
    )

    if result.szs_status not in success:
        outcome = "unknown" if result.szs_status is None else result.szs_status.value
        return None, outcome
    # Only successful searches need the matrix again for the training graph.
    # Keeping the proof state's matrix alive during search roughly doubled the
    # footprint of every timeout and failed problem. Both use the same
    # clausification, so SAT-core clause IDs index the graph's clauses.

    loaded = load_tptp_problem(problem, tptp_root = tptp_root)
    try:
        core = sat_core_clause_ids(agent.diagnostics(), matrix_size = len(loaded.matrix))
    except ValueError as error:
        return None, f"invalid SAT core: {error}"

    check_sat_core_clauses(
        loaded.matrix,
        core,
        agent.diagnostics().get("sat_core_clause_texts"),
    )

    return training_example_from_sat_core(
        loaded.matrix,
        problem_path = loaded.requested_path,
        axiom_clause_ids = loaded.axiom_clause_ids,
        conjecture_clause_ids = loaded.conjecture_clause_ids,
        core_clause_ids = core,
    ), "proved"

def check_sat_core_clauses(matrix: Matrix, core: list[int], texts: object):
    if not isinstance(texts, list) or len(texts) != len(core):
        raise RuntimeError("SAT result is missing the clause texts of its core")

    for index, text in zip(core, texts, strict = True):
        if str(matrix.clauses[index]) != text:
            raise RuntimeError(
                f"SAT-core clause {index} is {text!r} in the proof matrix but {str(matrix.clauses[index])!r} in the labelled matrix"
            )
