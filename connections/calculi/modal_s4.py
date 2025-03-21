from typing import Any, List, Tuple

from connections.calculi.intuitionistic import IConnectionState
from connections.utils.primitives import Function


class S4ConnectionState(IConnectionState):
    """State representation for S4 modal connection calculus.

    Extends the intuitionistic connection calculus with S4-specific prefix handling.
    The S4 modal logic requires reflexive and transitive accessibility relations
    between worlds.

    This implementation supports constant, cumulative, and varying domains.
    For cumulative domains, the prefix unification behavior is the same as
    intuitionistic logic. For varying domains, full prefix unification is required.

    Unlike intuitionistic logic, S4 does not append fresh variables to prefixes
    during unification.
    """

    def _pre_eq(self, lit_1: Any, lit_2: Any) -> Tuple[Function, Function]:
        """Prepare prefixes for unification between two literals in S4.

        In S4, the entire prefix is relevant for unification, but unlike
        intuitionistic logic, no fresh variables are appended to the prefixes.

        Args:
            lit_1: First literal.
            lit_2: Second literal.

        Returns:
            Tuple of the full prefixes of both literals.
        """
        if lit_2.prefix is None:
            lit_2.prefix = Function("string")
        if lit_1.prefix is None:
            lit_1.prefix = Function("string")
        pre_1, pre_2 = lit_1.prefix, lit_2.prefix
        return pre_1, pre_2

    def _admissible_pairs(self) -> List[Tuple[Function, Function]]:
        """Generate admissible prefix pairs based on domain type in S4.

        The prefix unification requirements vary by domain type:
        - Constant domains: No prefix unification needed
        - Cumulative domains: Same as intuitionistic logic
        - Varying domains: Full prefix unification required

        Returns:
            List of prefix pairs that need to be unified.
        """
        if self.settings.domain in "constant":
            return []
        if self.settings.domain == "cumulative":
            return super()._admissible_pairs()
        if self.settings.domain == "varying":
            equations = []
            for var, term in self.substitution.to_dict().items():
                for eigen in self._find_eigenvariables(term):
                    equations.append((var.prefix, eigen.prefix))
            return equations
