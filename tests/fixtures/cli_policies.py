from __future__ import annotations

import time

from connections.policy import Policy
from connections.prover.state import State


class BlockingPolicy(Policy):
    def __init__(self, **_: object) -> None:
        pass

    def __call__(self, state: State) -> object:
        _ = state
        time.sleep(10)
        return None


class NotAPolicy:
    pass
