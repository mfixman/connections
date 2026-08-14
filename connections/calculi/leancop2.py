"""Explicit-state implementation of the leanCoP 2.x core search rules.

The surrounding project exposes proof search as actions rather than relying on
Prolog's call stack.  This state class supplies the leanCoP details that are
missing from :mod:`classicalcut`: scoped lemmata, precise regularity, ground
extensions beyond the path bound, restricted starts, and leanCoP's controlled
iterative deepening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from connections.calculi.classicalcut import (
    ClassicalCut,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    Start,
)
from connections.utils.primitives import Expression, Literal, Variable
from connections.utils.tableau import Tableau


@dataclass
class Lemma(ConnectionAction):
    """Close the current literal using an earlier solved sibling literal."""

    lemma_lit: Literal | None = None
    principle_node: Tableau | None = None

    def __repr__(self) -> str:
        suffix = " [cut]" if self.cut else ""
        return f"Lemma {self.id}: {self.lemma_lit}{suffix}"


@dataclass
class Finish(ConnectionAction):
    """Report exhaustive failure of the final complete search."""


def _is_ground_expression(expression: Expression) -> bool:
    if isinstance(expression, Variable):
        return False
    return all(_is_ground_expression(argument) for argument in expression.args)


class LeanCoP2State(ClassicalCut):
    """Classical connection state with the leanCoP 2.1 core optimisations."""

    def reset(self, depth: int | None = None) -> None:
        self.path_limit_hit = False
        self.restricted_backtracking = bool(self.settings.leancop_cut)
        self.restricted_start = bool(self.settings.leancop_scut)
        self.comp_limit = self.settings.leancop_comp_limit
        self.conjecture_start = bool(self.settings.leancop_conjecture_start)
        super().reset(depth)

    def _conjecture_start_indices(self) -> list[int]:
        return [
            index
            for index, clause in enumerate(self.matrix.clauses)
            if any(literal.symbol == '#' and not literal.neg for literal in clause)
        ]

    def _uses_conjecture_markers(self) -> bool:
        # This is the exact test in leanCoP's prove2/3.  The ``conj`` option
        # influences clausification, but start selection itself depends on
        # whether the resulting matrix contains the administrative [-#]
        # clause.  Without it, Prolog falls back to positive starts.
        return any(
            len(clause) == 1 and clause[0].symbol == '#' and clause[0].neg
            for clause in self.matrix.clauses
        )

    def starts(self) -> list[Start]:
        candidates = (
            self._conjecture_start_indices()
            if self._uses_conjecture_markers()
            else list(self.matrix.positive_clauses)
        )
        if self.restricted_start:
            candidates = candidates[:1]

        starts: list[Start] = []
        for clause_index in candidates:
            clause_copy = [
                literal
                for literal in self.matrix.copy(clause_index)
                if not (literal.symbol == '#' and not literal.neg)
            ]
            starts.append(
                Start(
                    id=f'st{len(starts)}',
                    clause_copy=clause_copy,
                )
            )
        return starts

    def _current_subgoal_clause(self) -> list[Literal]:
        """Return leanCoP's current ``[Lit|Cla]``, excluding solved siblings."""
        siblings = self.goal.parent.children
        current_index = siblings.index(self.goal)
        return [node.literal for node in siblings[current_index:]]

    def _lemma_literals(self, node) -> list[Literal]:
        """Reconstruct the scoped ``Lem`` argument of leanCoP's prove/5."""
        parent = node.parent
        if parent is None:
            return []

        inherited = [] if parent.literal is None else self._lemma_literals(parent)
        current_index = parent.children.index(node)
        first_clause_index = 0 if parent.literal is None else 1
        solved_siblings = [
            sibling.literal
            for sibling in parent.children[first_clause_index:current_index]
            if sibling.proven and sibling.literal is not None
        ]
        # Prolog conses each newly solved literal onto the front of Lem.
        return list(reversed(solved_siblings)) + inherited

    def lemmas(self) -> list[Lemma]:
        for lemma_lit in self._lemma_literals(self.goal):
            # Prolog's ``Lit == LitL`` compares complete literals.  The
            # generic substitution equality intentionally compares only the
            # atom and arguments, because it is also used after independent
            # polarity checks by regularity.  Check polarity explicitly here:
            # a solved p(X) sibling is not a lemma for a later -p(X) goal.
            if (
                self.goal.literal.neg == lemma_lit.neg
                and self.substitution.equal(self.goal.literal, lemma_lit)
            ):
                return [
                    Lemma(
                        id='lm0',
                        lemma_lit=lemma_lit,
                        principle_node=self.goal,
                    )
                ]
        return []

    def reductions(self) -> list[Reduction]:
        reductions = super().reductions()
        # leanCoP starts prove/5 with the administrative literal -# on
        # Path.  Direct Start actions avoid materialising that extra tableau
        # node, so provide its one possible reduction explicitly.
        if self.goal.literal.symbol == '#' and not self.goal.literal.neg:
            reductions.insert(
                0,
                Reduction(
                    id='re_marker',
                    principle_node=self.goal,
                    sub_updates=[],
                    path_lit=Literal('#', [], neg=True),
                ),
            )
        return reductions

    @staticmethod
    def _ground_extension(extension: Extension) -> bool:
        return all(_is_ground_expression(literal) for literal in extension.clause_copy)

    def legal_forward_actions(self) -> dict[str, ConnectionAction]:
        current_clause = self._current_subgoal_clause()
        virtual_marker_repeats = any(
            literal.symbol == '#' and literal.neg for literal in current_clause
        )
        if self.goal is None or virtual_marker_repeats or self.regularizable(current_clause):
            actions: list[ConnectionAction] = []
        else:
            extensions = self.extensions()
            actions = cast(
                list[ConnectionAction],
                self.lemmas() + self.reductions(),
            )
            # The administrative -# literal is the first member of leanCoP's
            # active Path after the start step, hence the explicit tableau's
            # path length is goal.depth + 1.
            if self.goal.depth + 1 < self.max_depth:
                # Prolog keeps assertion order here; groundness only decides
                # whether an extension survives at the path limit.  Reordering
                # ground alternatives first changes cut behavior.
                actions.extend(extensions)
            else:
                for extension in extensions:
                    if self._ground_extension(extension):
                        actions.append(extension)
                    else:
                        self.path_limit_hit = True

        if self.settings.connection_constraints:
            actions = [action for action in actions if not self.violates_constraint(action)]
        return {action.id: action for action in actions}

    def resets(self) -> list[Reset]:
        # comp(J) deliberately deepens even when an incomplete cut search did
        # not touch the path bound.  Other phases deepen only after PathLim was
        # actually exceeded, exactly as leanCoP's pathlim/0 flag does.
        if self.comp_limit is not None or self.path_limit_hit:
            return [Reset(id='rs', depth=self.max_depth + 1)]
        return []

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            return {action.id: action for action in self.starts()}
        return self.legal_forward_actions()

    def control_actions(self) -> list[ConnectionAction]:
        if self.goal.parent is None:
            return cast(list[ConnectionAction], self.resets())
        return cast(list[ConnectionAction], self.backtracks() + self.resets())

    def update_goal(self, action: ConnectionAction) -> None:
        if isinstance(action, Finish):
            self.info = 'Non-Theorem: exhaustive complete search'
            self.is_terminal = True
            return

        if isinstance(action, Lemma):
            if action.id in self.goal.actions:
                del self.goal.actions[action.id]
            self.goal.num_attempted += 1
            # Maintain one substitution trail frame per proof action so the
            # inherited explicit backtracker remains balanced.
            self.substitution.update([])
            self.proof_sequence.append(action)
            self.goal.proven = True
            if action.cut:
                self.goal.cut_committed = True
            self.theorem_or_next()
            return

        super().update_goal(action)
