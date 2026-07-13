"""TPTP certificates for the packaged SATResetCoP strategy.

SATResetCoP's decisive object is an UNSAT collection of grounded shadow
clauses.  For FOF input, leanCoP's translator constructs a DNF connection
matrix for ``axioms => conjecture``.  Complementing every matrix path gives a
CNF equisatisfiable with ``axioms & ~conjecture``; complementing all shadow
clauses preserves (un)satisfiability.  The emitted certificate therefore
documents:

1. the original input formulae and negated conjecture;
2. the equisatisfiable FOF-to-CNF connection-matrix translation;
3. the ground instances selected by SATResetCoP; and
4. the ground SAT refutation.

This is intentionally specific to the CASC ``SATResetCoP`` configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from connections.calculi.classicalsat import SATClause, SATConnectionState
from connections.utils.classical_problem_parsing import (
    _parse_clause as _parse_tptp_clause,
    _parse_name_list,
    _resolve_include,
    _split_top_level,
    _split_tptp_statements,
    _strip_tptp_comments,
)
from connections.utils.primitives import Expression, Literal, Variable


class CertificateError(RuntimeError):
    """Raised when a theorem cannot be backed by a complete certificate."""


@dataclass(frozen=True)
class _InputFormula:
    language: str
    original_name: str
    role: str
    formula: str
    source_file: str


def render_satresetcop_certificate(problem: str, state: SATConnectionState) -> str:
    """Return a complete TPTP CNF-refutation certificate.

    Raw untyped FOF is the intended CASC use. Raw CNF is also handled so the
    StarExec configuration remains safe when it is smoke-tested on a CNF
    problem.
    """
    problem_path = Path(problem).resolve()
    inputs = _collect_input_formulae(problem_path)
    if not inputs:
        raise CertificateError('certificate input has no FOF or CNF formulae')

    core = state.certificate_ground_clauses()
    fof_input = any(item.language == 'fof' for item in inputs)
    if fof_input:
        lines, matrix_clauses = _render_fof_prefix(inputs, state)
    else:
        lines, matrix_clauses = _render_cnf_prefix(inputs, state)

    complement = fof_input
    ground_names: list[str] = []
    for index, clause in enumerate(core, start=1):
        origin = state.shadow.clause_origins.get(clause)
        expanded = state.shadow.expanded_clauses.get(clause)
        if origin is None or not 0 <= origin < len(matrix_clauses):
            raise CertificateError(f'ground clause {index} has no matrix origin')
        if expanded is None:
            raise CertificateError(f'ground clause {index} has no printable instance')

        name = f'satresetcop_ground_{index}'
        ground_names.append(name)
        # Grounding a clause to the fresh constant satresetcop_ground is
        # universal instantiation (a sound thm step), not Skolemization, so it
        # carries no new_symbols annotation.
        lines.extend(_annotated_formula(
            'cnf',
            name,
            'plain',
            _render_ground_clause(expanded, complement=complement),
            f'inference(instantiate,[status(thm)],[satresetcop_matrix_{origin + 1}])',
        ))

    lines.extend(_annotated_formula(
        'cnf',
        'satresetcop_false',
        'plain',
        '$false',
        'inference(sat_refutation,[status(thm)],[' + ','.join(ground_names) + '])',
        useful_info='[proof]',
    ))
    return '\n'.join(lines)


def _render_fof_prefix(
    inputs: list[_InputFormula],
    state: SATConnectionState,
) -> tuple[list[str], list[str]]:
    if not any(item.role == 'conjecture' for item in inputs):
        raise CertificateError('FOF certificate input has no conjecture')

    lines: list[str] = []
    premise_parents: list[str] = []

    for index, item in enumerate(inputs, start=1):
        leaf_name = f'satresetcop_input_{index}'
        lines.extend(_annotated_formula(
            item.language,
            leaf_name,
            item.role,
            item.formula,
            f"file({_quote(item.source_file)},{item.original_name})",
        ))

        if item.role == 'conjecture':
            negated_name = f'satresetcop_negated_conjecture_{index}'
            negated_formula = f'~ ({item.formula})'
            lines.extend(_annotated_formula(
                'fof',
                negated_name,
                'negated_conjecture',
                negated_formula,
                f'inference(negate,[status(cth)],[{leaf_name}])',
            ))
            premise_parents.append(negated_name)
        else:
            premise_parents.append(leaf_name)

    matrix_clauses = [
        _render_matrix_clause(clause, complement=True)
        for clause in state.matrix.clauses
    ]
    if not matrix_clauses:
        raise CertificateError('FOF translation produced an empty connection matrix')

    translation_info = ['status(esa)', *_matrix_generated_symbol_info(state)]
    lines.extend(_annotated_formula(
        'fof',
        'satresetcop_cnf_matrix',
        'plain',
        _universally_closed_conjunction(matrix_clauses),
        (
            'inference(clausify,['
            + ','.join(translation_info)
            + '],[' + ','.join(premise_parents) + '])'
        ),
    ))

    for index, clause_text in enumerate(matrix_clauses, start=1):
        lines.extend(_annotated_formula(
            'cnf',
            f'satresetcop_matrix_{index}',
            'plain',
            clause_text,
            'inference(split_conjunct,[status(thm)],[satresetcop_cnf_matrix])',
        ))
    return lines, matrix_clauses


def _render_cnf_prefix(
    inputs: list[_InputFormula],
    state: SATConnectionState,
) -> tuple[list[str], list[str]]:
    if any(item.language != 'cnf' for item in inputs):
        raise CertificateError('mixed FOF/CNF input requires the FOF translation path')

    lines: list[str] = []
    active_leaf_names: list[str] = []
    for index, item in enumerate(inputs, start=1):
        leaf_name = f'satresetcop_input_{index}'
        lines.extend(_annotated_formula(
            item.language,
            leaf_name,
            item.role,
            item.formula,
            f"file({_quote(item.source_file)},{item.original_name})",
        ))
        if _parse_tptp_clause(item.formula) is not None:
            active_leaf_names.append(leaf_name)

    matrix_clauses = [
        _render_matrix_clause(clause, complement=False)
        for clause in state.matrix.clauses
    ]
    if len(matrix_clauses) != len(active_leaf_names):
        raise CertificateError(
            'raw CNF leaves do not correspond one-to-one with the parsed matrix'
        )
    for index, (clause_text, leaf_name) in enumerate(
        zip(matrix_clauses, active_leaf_names),
        start=1,
    ):
        lines.extend(_annotated_formula(
            'cnf',
            f'satresetcop_matrix_{index}',
            'plain',
            clause_text,
            f'inference(normalize,[status(thm)],[{leaf_name}])',
        ))
    return lines, matrix_clauses


def _collect_input_formulae(problem: Path) -> list[_InputFormula]:
    collected: list[_InputFormula] = []
    seen: set[tuple[Path, str]] = set()

    def visit(path: Path, display_name: str, selected: set[str] | None = None) -> None:
        text = _strip_tptp_comments(path.read_text(errors='replace'))
        for statement in _split_tptp_statements(text):
            if statement.startswith('include('):
                parts = _split_top_level(statement[len('include('):-1], ',')
                include_text = parts[0].strip().strip("'")
                include_path = _resolve_include(parts[0], path)
                include_names = _parse_name_list(parts[1]) if len(parts) > 1 else None
                visit(include_path, include_text, include_names)
                continue

            match = re.match(r'^(fof|cnf)\s*\((.*)\)$', statement, flags=re.DOTALL)
            if match is None:
                continue
            language, body = match.groups()
            parts = _split_top_level(body, ',')
            if len(parts) < 3:
                raise CertificateError(f'malformed {language} input statement')
            original_name = parts[0].strip()
            if selected is not None and original_name not in selected:
                continue
            key = (path.resolve(), original_name)
            if key in seen:
                continue
            seen.add(key)
            collected.append(_InputFormula(
                language=language,
                original_name=original_name,
                role=parts[1].strip(),
                formula=parts[2].strip(),
                source_file=display_name,
            ))

    visit(problem, _problem_source_name(problem))
    return collected


def _problem_source_name(problem: Path) -> str:
    try:
        text = problem.read_text(errors='replace')
    except OSError:
        return problem.name
    declared = re.search(
        r'^\s*%\s*File\s*:\s*([^\s:]+)',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if declared is None:
        return problem.name
    name = declared.group(1)
    return name if name.endswith(('.p', '.ax')) else f'{name}.p'


def _annotated_formula(
    language: str,
    name: str,
    role: str,
    formula: str,
    source: str,
    *,
    useful_info: str | None = None,
) -> list[str]:
    result = [f'{language}({name},{role},', f'    {formula},', f'    {source}']
    if useful_info is not None:
        result[-1] += ','
        result.append(f'    {useful_info}')
    result[-1] += ').'
    return result


def _conjunction(formulae: list[str]) -> str:
    if not formulae:
        return '$true'
    if len(formulae) == 1:
        return f'({formulae[0]})'
    return '( ' + '\n    & '.join(f'({formula})' for formula in formulae) + ' )'


def _universally_closed_conjunction(formulae: list[str]) -> str:
    conjunction = _conjunction(formulae)
    variables = sorted(
        set(re.findall(r'\bX[0-9]+\b', conjunction)),
        key=lambda variable: int(variable[1:]),
    )
    if not variables:
        return conjunction
    return '! [' + ','.join(variables) + '] :\n    ' + conjunction


def _render_matrix_clause(clause: list[Literal], *, complement: bool) -> str:
    variables: dict[tuple[str, int], str] = {}
    literals = [
        _render_literal(literal, variables, complement=complement)
        for literal in clause
    ]
    if not literals:
        return '$false'
    return '( ' + ' | '.join(literals) + ' )'


def _render_ground_clause(expanded: list[str], *, complement: bool) -> str:
    literals = []
    for literal in expanded:
        negative = literal.startswith('-')
        atom = literal[1:] if negative else literal
        atom = _normalize_generated_symbol_text(atom)
        atom = atom.replace('__ground__', 'satresetcop_ground')
        if complement:
            negative = not negative
        atom = _normalize_ground_equality(atom)
        literals.append(f'~ {atom}' if negative else atom)
    if not literals:
        return '$false'
    return '( ' + ' | '.join(literals) + ' )'


def _render_literal(
    literal: Literal,
    variables: dict[tuple[str, int], str],
    *,
    complement: bool,
) -> str:
    symbol, args = _normalize_indexed_symbol(literal.symbol, literal.args)
    negative = not literal.neg if complement else literal.neg
    if symbol == '=' and len(args) == 2:
        atom = (
            '( '
            + _render_term(args[0], variables)
            + ' = '
            + _render_term(args[1], variables)
            + ' )'
        )
        return f'~ {atom}' if negative else atom

    atom = _render_symbol(symbol)
    if args:
        atom += '(' + ','.join(_render_term(arg, variables) for arg in args) + ')'
    return f'~ {atom}' if negative else atom


def _render_term(term: Expression, variables: dict[tuple[str, int], str]) -> str:
    if isinstance(term, Variable):
        key = (term.symbol, term.copy_num)
        if key not in variables:
            variables[key] = f'X{len(variables) + 1}'
        return variables[key]
    raw_symbol, args = _normalize_indexed_symbol(str(term.symbol), term.args)
    symbol = _render_symbol(raw_symbol)
    if not args:
        return symbol
    return symbol + '(' + ','.join(_render_term(arg, variables) for arg in args) + ')'


def _matrix_generated_symbol_info(state: SATConnectionState) -> list[str]:
    skolem: set[str] = set()
    definitions: set[str] = set()
    predicates: set[str] = set()

    def visit_term(term: Expression) -> None:
        if isinstance(term, Variable):
            return
        symbol, args = _normalize_indexed_symbol(str(term.symbol), term.args)
        if symbol.startswith('f_skolem'):
            skolem.add(symbol)
        for arg in args:
            visit_term(arg)

    for clause in state.matrix.clauses:
        for literal in clause:
            symbol, args = _normalize_indexed_symbol(literal.symbol, literal.args)
            if symbol.startswith('p_defini'):
                definitions.add(symbol)
            if symbol == 'equal___':
                predicates.add(symbol)
            for arg in args:
                visit_term(arg)

    info = []
    if skolem:
        info.append(
            'new_symbols(skolem,['
            + ','.join(map(_render_symbol, sorted(skolem)))
            + '])'
        )
    if definitions:
        info.append(
            'new_symbols(definition,['
            + ','.join(map(_render_symbol, sorted(definitions)))
            + '])'
        )
    if predicates:
        info.append(
            'new_symbols(predicate,['
            + ','.join(map(_render_symbol, sorted(predicates)))
            + '])'
        )
    return info


def _normalize_indexed_symbol(
    symbol: str,
    args: list[Expression],
) -> tuple[str, list[Expression]]:
    """Use conventional names for leanCoP's indexed generated symbols.

    The translator internally encodes its first Skolem function as
    ``f_skolem(1,...)`` (and likewise for definitional predicates).  Some TPTP
    derivation checkers reject a numeric tag in that position.  In the proof
    only, render the equivalent fresh symbols ``f_skolem_1(...)`` and
    ``p_defini_1(...)``.
    """
    if symbol not in {'f_skolem', 'p_defini'} or not args:
        return symbol, args
    index = args[0]
    if isinstance(index, Variable) or index.args:
        return symbol, args
    index_text = str(index.symbol)
    if not index_text.isdigit():
        return symbol, args
    return f'{symbol}_{index_text}', args[1:]


def _normalize_generated_symbol_text(text: str) -> str:
    for symbol in ('f_skolem', 'p_defini'):
        text = re.sub(
            rf'\b{symbol}\(\s*([0-9]+)\s*,\s*',
            lambda match: f'{symbol}_{match.group(1)}(',
            text,
        )
        text = re.sub(
            rf'\b{symbol}\(\s*([0-9]+)\s*\)',
            lambda match: f'{symbol}_{match.group(1)}',
            text,
        )
    return text


def _normalize_ground_equality(text: str) -> str:
    text = text.strip()
    if not (text.startswith('=(') and text.endswith(')')):
        return text
    args = _split_top_level(text[2:-1], ',')
    if len(args) != 2:
        return text
    return f'( {args[0].strip()} = {args[1].strip()} )'


def _render_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith("'") and symbol.endswith("'"):
        return symbol
    if (
        re.fullmatch(r'[a-z][A-Za-z0-9_]*', symbol)
        or re.fullmatch(r'\$[A-Za-z0-9_]+', symbol)
    ):
        return symbol
    if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?', symbol):
        return symbol
    return _quote(symbol)


def _quote(text: str) -> str:
    return "'" + text.replace('\\', '\\\\').replace("'", "\\'") + "'"
