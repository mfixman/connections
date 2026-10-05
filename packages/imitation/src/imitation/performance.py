"""Use model predictions to choose among a proof-search agent's available actions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from connections.agent import Agent, AgentOptions, AgentStatus, Chooser
from connections.environment.actions import Action, UndoAction
from connections.environment.dynamics import Dynamics
from connections.environment.state import State

from imitation.representation.preprocess import GraphPreprocessor
from imitation.representation.schema import GraphInput


class ActionModel(Protocol):
    """Scores one decision: a graph input in, an index into its rows out."""

    def __call__(self, model_input: GraphInput) -> int: ...


@dataclass(frozen=True, slots=True)
class ModelChooser:
    """Score candidate actions; skip inference when only one action is available."""

    preprocess: GraphPreprocessor
    model: ActionModel

    def __call__(self, state: State, actions: Sequence[Action]) -> Action:
        if len(actions) == 1:
            return actions[0]
        index = self.model(self.preprocess(state, actions))
        if not 0 <= index < len(actions):
            raise IndexError(
                f"model chose index {index} among {len(actions)} candidates"
            )
        return actions[index]


class AllActionsMarkovAgent(Agent):
    """The memoryless agent over the whole surface A(s).

    Candidates are every apply action of every fringe goal, in fringe and
    rule order, then one undo per standing rule application, newest first.
    Backtracking is a candidate like any other, which is what lets a learned
    chooser predict it; the agent itself keeps no memory at all.
    """

    def __init__(self, choose: Chooser, options: AgentOptions | None = None) -> None:
        super().__init__(options)
        self.choose = choose

    def __call__(self, state: State) -> Action | None:
        if state.tableau.root.closed:
            self.status = AgentStatus.CLOSED
            return None
        apply_actions = tuple(
            action
            for goal in state.fringe
            for action in Dynamics.apply_actions(
                state,
                goal,
                factorization=self.options.factorization,
                start=self.options.start,
            ).ordered()
        )
        undo_actions = tuple(
            UndoAction(application_id)
            for application_id in sorted(state.tableau.rule_applications, reverse=True)
        )
        actions = apply_actions + undo_actions
        if not actions:
            self.status = AgentStatus.GAVE_UP
            return None
        self.status = AgentStatus.SEARCHING
        return self.choose(state, actions)


@dataclass(frozen=True, slots=True)
class PerformanceRecipe:
    """Build acting and replay agents with the same options and preprocessor.

    The surface key excludes initial_depth because replay adjusts it to cover the proof.
    """

    agent_class: Callable[..., Agent]
    options: AgentOptions = AgentOptions()
    preprocess: GraphPreprocessor = GraphPreprocessor()

    def __post_init__(self) -> None:
        if self.options.start != self.preprocess.start:
            raise ValueError(
                "the agent's start mode and the preprocessor's must agree: "
                f"{self.options.start!r} != {self.preprocess.start!r}"
            )

    def with_chooser(
        self, choose: Chooser, *, initial_depth: int | None = None
    ) -> Agent:
        options = self.options
        if initial_depth is not None:
            options = replace(options, initial_depth=initial_depth)
        return self.agent_class(choose, options)

    def surface_key(self) -> str:
        options = self.options
        return "/".join(
            (
                getattr(self.agent_class, "__qualname__", repr(self.agent_class)),
                f"cut={options.cut}",
                f"scut={options.scut}",
                f"comp={options.comp}",
                f"backtrack={options.backtrack}",
                f"factorization={options.factorization}",
                f"start={options.start}",
                f"{self.preprocess.name}.v{self.preprocess.version}",
            )
        )


__all__ = [
    "ActionModel",
    "AllActionsMarkovAgent",
    "ModelChooser",
    "PerformanceRecipe",
]
