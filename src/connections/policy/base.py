from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, TypeAlias

from connections.prover.actions import Action
from connections.prover.state import State


PolicyDecision: TypeAlias = Action | None
BacktrackGranularity: TypeAlias = Literal["step", "maximal"]


class Policy(ABC):
    """Action policy used by ``Prover``.

    The only public policy operation is calling it with the current prover
    state. Plain policies return an action or ``None``. Search policies such as
    DFS and ID may also return a prover outcome when their search is exhausted.
    """

    @abstractmethod
    def __call__(self, state: State) -> object:
        raise NotImplementedError

    def accepts_tableau_proof(self, state: State) -> bool:
        """Return whether an ordinary closed tableau is terminal for this policy."""
        _ = state
        return True

    def diagnostics(self) -> Mapping[str, object]:
        """Return JSON-serializable diagnostics collected during this run."""
        return {}


__all__ = [
    "BacktrackGranularity",
    "PolicyDecision",
    "Policy",
]
