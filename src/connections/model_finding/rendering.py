from __future__ import annotations

import re

from connections.model_finding.core import FiniteModel


_PLAIN_SYMBOL = re.compile(r"^[a-z][A-Za-z0-9_]*$")


def render_tptp_model(model: FiniteModel) -> str:
    """Render a total model using the TPTP finite-interpretation roles."""

    elements = tuple(f"model_d{value}" for value in model.domain)
    closure = " | ".join(f"X = {element}" for element in elements)
    domain_parts = [f"! [X] : ({closure})"]
    for left_index, left in enumerate(elements):
        for right in elements[left_index + 1 :]:
            domain_parts.append(f"{left} != {right}")

    function_parts: list[str] = []
    for symbol, table in sorted(model.functions.items()):
        name = _quote_symbol(symbol)
        for arguments, output in sorted(table.items()):
            rendered_arguments = tuple(elements[value] for value in arguments)
            application = name if not arguments else f"{name}({','.join(rendered_arguments)})"
            function_parts.append(f"{application} = {elements[output]}")

    predicate_parts: list[str] = []
    for symbol, table in sorted(model.predicates.items()):
        name = _quote_symbol(symbol)
        for arguments, truth in sorted(table.items()):
            rendered_arguments = tuple(elements[value] for value in arguments)
            application = name if not arguments else f"{name}({','.join(rendered_arguments)})"
            value_formula = application if truth else f"~({application})"
            predicate_parts.append(value_formula)

    return "\n".join(
        (
            _interpretation_statement("model_domain", "fi_domain", domain_parts),
            _interpretation_statement("model_functors", "fi_functors", function_parts or ["$true"]),
            _interpretation_statement(
                "model_predicates", "fi_predicates", predicate_parts or ["$true"]
            ),
            "",
        )
    )


def _interpretation_statement(name: str, role: str, parts: list[str]) -> str:
    body = "\n    & ".join(f"({part})" for part in parts)
    return f"fof({name},{role},(\n    {body}\n))."


def _quote_symbol(symbol: str) -> str:
    if _PLAIN_SYMBOL.fullmatch(symbol):
        return symbol
    return "'" + symbol.replace("\\", "\\\\").replace("'", "\\'") + "'"


__all__ = ["render_tptp_model"]
