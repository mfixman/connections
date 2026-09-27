from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import sys

DEEP_RECURSION_LIMIT = 1_000_000


@contextmanager
def deep_recursion(limit: int = DEEP_RECURSION_LIMIT) -> Iterator[None]:
    """Temporarily raise Python's recursion limit.

    TPTP problems with thousands of axioms parse into conjunction chains whose
    depth grows with the axiom count, and the clausifier and term walkers
    recurse over them. Python frames live on the heap, and C-level recursion
    keeps its own guard, so a high limit only fails on genuinely runaway
    recursion instead of on ordinary large problems.
    """

    previous = sys.getrecursionlimit()
    if previous >= limit:
        yield
        return
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


__all__ = ["DEEP_RECURSION_LIMIT", "deep_recursion"]
