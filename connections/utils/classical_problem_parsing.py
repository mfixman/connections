import os
import re
import subprocess
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

from connections.utils.cnf_parsing import file2cnf as parse_connections_cnf
from connections.utils.primitives import Constant, Function, Literal, Matrix, ProblemMetadata, Variable

_TYPED_SPC_FAMILIES = frozenset({'TCF', 'TF0', 'TF1', 'TX0', 'TX1', 'TH0', 'TH1', 'NX0', 'NH0'})
_UNSAT_STATUSES = frozenset({'Theorem', 'Unsatisfiable', 'ContradictoryAxioms'})
_SAT_STATUSES = frozenset({'Satisfiable', 'CounterSatisfiable', 'Open'})
_UNKNOWN_STATUSES = frozenset({'Unknown'})
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSICAL_TRANSLATION_DIR = _REPO_ROOT / 'translation' / 'classical'

def file2cnf(path: str):
    """Load a classical problem from internal CNF or raw TPTP syntax."""
    file_path = Path(path)
    text = file_path.read_text()

    match _problem_kind(file_path, text):
        case 'connections':
            return parse_connections_cnf(path)
        case 'tptp-cnf':
            metadata = problem_metadata(file_path)
            if metadata.typed:
                _fail_for_typed_problem(file_path, metadata)
            matrix = _parse_tptp_cnf(file_path)
            matrix.metadata = metadata
            return matrix
        case 'tptp-fof':
            metadata = problem_metadata(file_path)
            if metadata.typed:
                _fail_for_typed_problem(file_path, metadata)
            return _translate_tptp_problem(file_path, metadata)
        case _:
            _fail_for_unsupported_problem(file_path)

def _problem_kind(path: Path, text: str) -> str:
    """Classify the file so the caller can choose the right loader."""
    if _is_connections_cnf(text):
        return 'connections'
    if _has_supported_tptp_suffix(path):
        if '+' in path.name or is_mptp_fof_problem_path(path):
            return 'tptp-fof'
        if '-' in path.name:
            return 'tptp-cnf'
    return 'unsupported'

def _has_supported_tptp_suffix(path: Path) -> bool:
    """Accept only normal top-level TPTP files for raw problem parsing."""
    return ''.join(path.suffixes) in {'.p', '.ax'}

def is_mptp_fof_problem_path(path: Path) -> bool:
    """Return whether *path* looks like an MPTP FOF problem filename."""
    if ''.join(path.suffixes) != '.p':
        return False

    parts = path.stem.split('__', maxsplit=1)
    if len(parts) != 2:
        return False

    article, formula = parts
    return bool(article and formula and '_' in formula)

def _fail_for_unsupported_problem(path: Path) -> None:
    """Warn and fail fast for inputs we do not intentionally support."""
    message = (
        f'unsupported problem file {path}: expected translated connection CNF '
        f'or a TPTP file with suffix .p/.ax and a "+", "-", or MPTP-style '
        f'"article__formula_article.p" filename'
    )
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)

def _fail_for_typed_problem(path: Path, metadata: ProblemMetadata) -> None:
    """Reject typed problems that the prover does not support semantically."""
    tags = ', '.join(sorted(metadata.spc_tags)) or 'none'
    message = f'unsupported typed problem file {path}: SPC tags={tags}'
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)

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
    clause_records = []
    text = _strip_tptp_comments(path.read_text())

    for statement in _split_tptp_statements(text):
        clause_records.extend(_parse_statement(statement, path, selected_names, seen))

    clauses = [clause for clause, _ in clause_records]
    goals = [
        index
        for index, (_, role) in enumerate(clause_records)
        if _is_goal_role(role)
    ]
    positives = [
        index
        for index, (clause, _) in enumerate(clause_records)
        if all(not literal.neg for literal in clause)
    ]
    has_non_goal = any(not _is_goal_role(role) for _, role in clause_records)
    start_clauses = positives if not goals or not has_non_goal else goals
    return Matrix(clauses, start_clauses=start_clauses)

def _parse_statement(
    statement: str,
    current_path: Path,
    selected_names: set[str] | None,
    seen: set[Path],
) -> list[tuple[list[Literal], str]]:
    """Dispatch a single top-level TPTP statement."""
    match statement:
        case s if s.startswith('include('):
            return _parse_include_statement(s, current_path, seen)
        case s if s.startswith('cnf('):
            return _parse_cnf_statement(s, selected_names)
        case s if s.startswith('tcf('):
            return _parse_tcf_statement(s, selected_names)
        case s if s.startswith(('fof(', 'tff(', 'thf(')):
            head = s.split('(', 1)[0]
            raise ValueError(
                f'unsupported {head} statement in raw CNF parser for {current_path}; '
                'use a supported untyped CNF problem or add a translation path'
            )
        case _:
            return []

def _parse_include_statement(statement: str, current_path: Path, seen: set[Path]) -> list[tuple[list[Literal], str]]:
    """Load clauses from an included TPTP file."""
    inner = statement[len('include('):-1]
    parts = _split_top_level(inner, ',')
    include_path = _resolve_include(parts[0], current_path)
    include_names = _parse_name_list(parts[1]) if len(parts) > 1 else None
    return [
        (clause, 'axiom')
        for clause in _parse_tptp_cnf(include_path, include_names, seen).clauses
    ]

def _parse_cnf_statement(statement: str, selected_names: set[str] | None) -> list[tuple[list[Literal], str]]:
    """Parse one `cnf(...)` declaration into zero or one clauses."""
    inner = statement[len('cnf('):-1]
    parts = _split_top_level(inner, ',')
    if len(parts) < 3:
        raise ValueError(f'malformed cnf statement: {statement[:80]}')

    name = parts[0].strip()
    if selected_names is not None and name not in selected_names:
        return []

    role = parts[1].strip()
    clause = _parse_clause(','.join(parts[2:]))
    if clause is None:
        return []
    return [(clause, role)]

def _parse_tcf_statement(statement: str, selected_names: set[str] | None) -> list[tuple[list[Literal], str]]:
    """Parse one `tcf(...)` declaration as a typed CNF clause."""
    inner = statement[len('tcf('):-1]
    parts = _split_top_level(inner, ',')
    if len(parts) < 3:
        raise ValueError(f'malformed tcf statement: {statement[:80]}')

    name = parts[0].strip()
    if selected_names is not None and name not in selected_names:
        return []

    role = parts[1].strip()
    clause = _parse_tcf_clause(','.join(parts[2:]))
    if clause is None:
        return []
    return [(clause, role)]

def _is_goal_role(role: str | None) -> bool:
    return role in {'conjecture', 'negated_conjecture'}

def _parse_tcf_clause(formula_text: str) -> list[Literal] | None:
    """Parse a typed CNF formula, discarding universal type prefixes."""
    formula_text = _strip_tcf_quantifiers(formula_text)
    return _parse_clause(formula_text)

def _strip_tcf_quantifiers(text: str) -> str:
    """Drop leading universal quantifiers from a TCF clause."""
    text = _strip_outer_parens(text)
    while text.startswith('!'):
        split_at = _top_level_operator(text, ':')
        if split_at < 0:
            raise ValueError(f'malformed typed quantifier: {text[:80]}')
        text = _strip_outer_parens(text[split_at + 1:])
    return text

def _preprocess_tptp_for_prolog(path: Path, workdir: Path) -> Path:
    """Quote ~| for SWI-Prolog 7+ which treats | as a meta-character.

    SWI-Prolog 7+ tokenizes ~| as two tokens (~ and |) instead of
    the single operator atom.  Quoting it as '~|' makes the reader
    produce the correct atom that matches the operator declaration.
    Returns *path* unchanged when no quoting is needed.
    """
    text = path.read_text()
    modified = re.sub(r"(?<!')~\|(?!')", "'~|'", text)
    if modified == text:
        return path
    rewritten_path = workdir / f'{path.stem}.quoted{path.suffix}'
    rewritten_path.write_text(modified, encoding = 'utf-8')
    return rewritten_path

def _translate_tptp_problem(path: Path, metadata: ProblemMetadata | None = None) -> Matrix:
    """Translate raw TPTP FOF to connection CNF, then parse it."""
    env = os.environ.copy()
    env.setdefault('TPTP', str(_tptp_root()))

    with tempfile.TemporaryDirectory(prefix = 'tptp-translate-') as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = _preprocess_tptp_for_prolog(path, temp_dir_path)
        output_path = temp_dir_path / f'{path.stem}.cnf'
        translator = _CLASSICAL_TRANSLATION_DIR / 'translate.sh'
        result = subprocess.run(
            ['sh', str(translator), str(input_path), str(output_path)],
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ValueError(detail or f'failed to translate {path}')
        matrix = parse_connections_cnf(str(output_path))
        if metadata is not None:
            matrix.metadata = metadata
        return matrix

@lru_cache(maxsize = None)
def problem_metadata(path: str | Path) -> ProblemMetadata:
    """Return cached metadata for a problem or included axiom file."""
    return _problem_metadata(Path(path).resolve())

def _problem_metadata(path: Path) -> ProblemMetadata:
    if not path.exists():
        return ProblemMetadata()

    text = path.read_text()
    declared_status = _parse_declared_status(text)
    declared_status_kind = _declared_status_kind(declared_status)
    spc_tags = _collect_spc_tags(path)
    typed = any(_spc_family(tag) in _TYPED_SPC_FAMILIES for tag in spc_tags)
    return ProblemMetadata(
        spc_tags = frozenset(spc_tags),
        typed = typed,
        declared_status = declared_status,
        declared_status_kind = declared_status_kind,
    )

def _parse_declared_status(text: str) -> str | None:
    match = re.search(r'^\s*%\s*Status\s*:\s*(.+?)\s*$', text, flags = re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()

def _declared_status_kind(status: str | None) -> str | None:
    if status is None:
        return None
    if status in _UNSAT_STATUSES:
        return 'unsat'
    if status in _SAT_STATUSES:
        return 'sat'
    if status in _UNKNOWN_STATUSES:
        return 'unknown'
    return 'other'

def _spc_family(tag: str) -> str:
    return tag.split('_', 1)[0]

def _collect_spc_tags(path: Path, seen: set[Path] | None = None) -> set[str]:
    """Collect SPC tags over a file's transitive include closure."""
    path = path.resolve()
    if seen is None:
        seen = set()
    if path in seen:
        return set()

    seen.add(path)
    text = path.read_text()
    tags = set(_parse_spc_tags(text))
    if not _has_supported_tptp_suffix(path):
        return tags

    for statement in _split_tptp_statements(_strip_tptp_comments(text)):
        if not statement.startswith('include('):
            continue
        inner = statement[len('include('):-1]
        parts = _split_top_level(inner, ',')
        include_path = _resolve_include(parts[0], path)
        tags.update(_collect_spc_tags(include_path, seen))
    return tags

def _parse_spc_tags(text: str) -> set[str]:
    return {
        match.group(1).strip()
        for match in re.finditer(r'^\s*%\s*SPC\s*:\s*(\S+)\s*$', text, flags = re.MULTILINE)
    }

def _strip_tptp_comments(text: str) -> str:
    return re.sub(r'%.*$', '', text, flags=re.MULTILINE)

def _repo_root() -> Path:
    """Return the source-tree root for include resolution."""
    return Path(__file__).resolve().parents[2]

def _tptp_root() -> Path:
    """Return the configured TPTP root from the environment."""
    env_root_raw = os.getenv('TPTP')
    if env_root_raw is not None:
        env_root = Path(env_root_raw).resolve()
        if env_root.is_dir():
            return env_root
        raise FileNotFoundError(f'TPTP is set but does not point to a directory: {env_root}')

    raise FileNotFoundError(
        'TPTP root is not configured; pass --tptp ROOT or set the TPTP environment variable'
    )

def _resolve_include(include_path: str, current_path: Path) -> Path:
    """Resolve an `include(...)` path relative to the file or the TPTP root."""
    include_path = include_path.strip().strip("'")

    for candidate in (current_path.parent / include_path, _tptp_root() / include_path):
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
