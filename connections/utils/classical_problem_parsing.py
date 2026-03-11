import os
import re
import subprocess
import tempfile
from pathlib import Path

from connections.utils.cnf_parsing import file2cnf as parse_connections_cnf
from connections.utils.primitives import Constant, Function, Literal, Matrix, Variable


def file2cnf(path: str):
    """Load a classical problem from internal CNF or raw TPTP syntax."""
    file_path = Path(path)
    text = file_path.read_text()

    match _problem_kind(file_path, text):
        case 'connections':
            return parse_connections_cnf(path)
        case 'tptp-cnf':
            return _parse_tptp_cnf(file_path)
        case 'tptp-fof':
            return _translate_tptp_problem(file_path)
        case _:
            return parse_connections_cnf(path)


def _problem_kind(path: Path, text: str) -> str:
    """Classify the file so the caller can choose the right loader."""
    if _is_connections_cnf(text):
        return 'connections'
    if '+' in path.name:
        return 'tptp-fof'
    if '-' in path.name:
        return 'tptp-cnf'
    return 'connections'


def _is_connections_cnf(text: str) -> bool:
    """Detect the repo's already-translated connection CNF format."""
    normalized = re.sub(r'^\s*%.*$', '', text, flags=re.MULTILINE)
    normalized = re.sub(r'\s+', '', normalized)
    return re.match(r'^cnf[i]?\([^,]+,[^,]+,\[', normalized) is not None


def _parse_tptp_cnf(
    path: Path,
    selected_names: set[str] | None = None,
    seen: set[Path] | None = None,
) -> Matrix:
    """Parse raw TPTP CNF, following local and repo-root includes."""
    path = path.resolve()
    if seen is None:
        seen = set()

    if path in seen:
        return Matrix([])

    seen.add(path)
    clauses = []
    text = re.sub(r'^\s*%.*$', '', path.read_text(), flags=re.MULTILINE)

    for statement in _split_tptp_statements(text):
        clauses.extend(_parse_statement(statement, path, selected_names, seen))

    return Matrix(clauses)


def _parse_statement(
    statement: str,
    current_path: Path,
    selected_names: set[str] | None,
    seen: set[Path],
) -> list[list[Literal]]:
    """Dispatch a single top-level TPTP statement."""
    match statement:
        case s if s.startswith('include('):
            return _parse_include_statement(s, current_path, seen)
        case s if s.startswith('cnf('):
            return _parse_cnf_statement(s, selected_names)
        case _:
            return []


def _parse_include_statement(statement: str, current_path: Path, seen: set[Path]) -> list[list[Literal]]:
    """Load clauses from an included TPTP file."""
    inner = statement[len('include('):-1]
    parts = _split_top_level(inner, ',')
    include_path = _resolve_include(parts[0], current_path)
    include_names = _parse_name_list(parts[1]) if len(parts) > 1 else None
    return _parse_tptp_cnf(include_path, include_names, seen).clauses


def _parse_cnf_statement(statement: str, selected_names: set[str] | None) -> list[list[Literal]]:
    """Parse one `cnf(...)` declaration into zero or one clauses."""
    inner = statement[len('cnf('):-1]
    parts = _split_top_level(inner, ',')
    if len(parts) < 3:
        raise ValueError(f'malformed cnf statement: {statement[:80]}')

    name = parts[0].strip()
    if selected_names is not None and name not in selected_names:
        return []

    clause = _parse_clause(','.join(parts[2:]))
    if clause is None:
        return []
    return [clause]


def _translate_tptp_problem(path: Path) -> Matrix:
    """Translate raw TPTP FOF to connection CNF, then parse it."""
    repo_root = _repo_root()
    translator = repo_root / 'translation' / 'classical' / 'translate.sh'
    env = os.environ.copy()
    env.setdefault('TPTP', str(_tptp_root()))

    with tempfile.NamedTemporaryFile(suffix='.cnf', delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [str(translator), str(path), str(output_path)],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ValueError(detail or f'failed to translate {path}')
        return parse_connections_cnf(str(output_path))
    finally:
        output_path.unlink(missing_ok=True)


def _repo_root() -> Path:
    """Return the repository root for translation and include resolution."""
    return Path(__file__).resolve().parents[2]


def _tptp_root() -> Path:
    """Use the repo root as the TPTP root when `Axioms/` is present."""
    repo_root = _repo_root()
    axioms = repo_root / 'Axioms'
    if axioms.is_dir():
        return repo_root
    raise FileNotFoundError(f'expected an Axioms directory at {axioms}')


def _resolve_include(include_path: str, current_path: Path) -> Path:
    """Resolve an `include(...)` path relative to the file or repo root."""
    include_path = include_path.strip().strip("'")

    for candidate in (current_path.parent / include_path, _repo_root() / include_path):
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'could not resolve include {include_path!r} from {current_path}')


def _parse_name_list(text: str) -> set[str] | None:
    """Parse the optional include name filter list."""
    text = _strip_outer_parens(text)
    if not text:
        return None
    if not (text.startswith('[') and text.endswith(']')):
        return None

    inner = text[1:-1].strip()
    if not inner:
        return set()
    return {part.strip() for part in _split_top_level(inner, ',')}


def _parse_clause(formula_text: str) -> list[Literal] | None:
    """Parse a CNF clause body, dropping tautological `$true` clauses."""
    formula_text = _strip_outer_parens(formula_text)

    if formula_text == '$true':
        return None
    if formula_text == '$false':
        return []

    literals = []
    for part in _split_top_level(formula_text, '|'):
        literal = _parse_literal(part)
        if type(literal) is bool:
            if literal:
                return None
            continue
        literals.append(literal)
    return literals


def _parse_literal(text: str) -> Literal | bool:
    """Parse one literal, preserving `$true` and `$false` as booleans."""
    text = _strip_outer_parens(text)

    if text.startswith('~'):
        literal = _parse_literal(_strip_outer_parens(text[1:]))
        if type(literal) is bool:
            if literal:
                return False
            return True
        literal.neg = not literal.neg
        return literal
    if text == '$true':
        return True
    if text == '$false':
        return False
    return _parse_atom(text)


def _parse_atom(text: str) -> Literal:
    """Parse a positive atomic formula or equality."""
    text = _strip_outer_parens(text)
    if not text:
        raise ValueError('empty atom')

    neq_idx = _top_level_operator(text, '!=')
    if neq_idx >= 0:
        left = _parse_term(text[:neq_idx])
        right = _parse_term(text[neq_idx + 2:])
        return Literal('=', [left, right], neg=True)

    eq_idx = _top_level_operator(text, '=')
    if eq_idx >= 0:
        left = _parse_term(text[:eq_idx])
        right = _parse_term(text[eq_idx + 1:])
        return Literal('=', [left, right])

    if '(' not in text:
        return Literal(text, [])

    name, rest = text.split('(', 1)
    args_text = rest[:-1]
    args = []
    if args_text:
        args = [_parse_term(part) for part in _split_top_level(args_text, ',')]
    return Literal(name, args)


def _parse_term(text: str):
    """Parse a TPTP term into the repo's term representation."""
    text = _strip_outer_parens(text)
    if not text:
        raise ValueError('empty term')

    if text[0].isupper() or text[0] == '_':
        return Variable(text)
    if text[0] in "'$":
        return Constant(text)
    if text[0].isdigit():
        return Constant(text)
    if '(' not in text:
        return Constant(text)

    name, rest = text.split('(', 1)
    args_text = rest[:-1]
    args = []
    if args_text:
        args = [_parse_term(part) for part in _split_top_level(args_text, ',')]
    return Function(name, args)


def _split_tptp_statements(text: str) -> list[str]:
    """Split a TPTP file into top-level statements ending in `.`."""
    statements = []
    depth = 0
    bracket_depth = 0
    in_quote = False
    start = 0

    for i, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        match char:
            case '(':
                depth += 1
            case ')':
                depth -= 1
            case '[':
                bracket_depth += 1
            case ']':
                bracket_depth -= 1
            case '.' if depth == 0 and bracket_depth == 0:
                statement = text[start:i].strip()
                if statement:
                    statements.append(statement)
                start = i + 1

    tail = text[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _split_top_level(text: str, delimiter: str) -> list[str]:
    """Split text on a delimiter that is not nested in brackets or quotes."""
    parts = []
    depth = 0
    bracket_depth = 0
    in_quote = False
    start = 0

    for i, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        match char:
            case '(':
                depth += 1
            case ')':
                depth -= 1
            case '[':
                bracket_depth += 1
            case ']':
                bracket_depth -= 1
            case _ if char == delimiter and depth == 0 and bracket_depth == 0:
                parts.append(text[start:i].strip())
                start = i + 1

    parts.append(text[start:].strip())
    return parts


def _strip_outer_parens(text: str) -> str:
    """Drop redundant outer parentheses from an expression string."""
    text = text.strip()
    while _has_outer_parens(text):
        text = text[1:-1].strip()
    return text


def _has_outer_parens(text: str) -> bool:
    """Check whether one pair of parentheses encloses the whole string."""
    if not (text.startswith('(') and text.endswith(')')):
        return False

    depth = 0
    in_quote = False
    for i, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return False

    return True


def _top_level_operator(text: str, operator: str) -> int:
    """Find an operator occurrence that is not nested in brackets or quotes."""
    depth = 0
    bracket_depth = 0
    in_quote = False

    for i, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        match char:
            case '(':
                depth += 1
            case ')':
                depth -= 1
            case '[':
                bracket_depth += 1
            case ']':
                bracket_depth -= 1

        if depth == 0 and bracket_depth == 0 and text.startswith(operator, i):
            return i

    return -1
