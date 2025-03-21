from typing import Any, List, Tuple

from connections.calculi.intuitionistic import IConnectionState
from connections.utils.primitives import Function


class S5ConnectionState(IConnectionState):
    """State representation for S5 modal connection calculus.

    Extends the intuitionistic connection calculus with S5-specific prefix handling.
    In S5, only the last world in the prefix is relevant for unification.

    This implementation supports constant, cumulative, and varying domains.
    For varying domains, additional prefix unification is required between
    variables and their substitutions.
    """

    def _pre_eq(self, lit_1: Any, lit_2: Any) -> Tuple[Function, Function]:
        """Prepare prefixes for unification between two literals in S5.

        In S5, only the last world in the prefix is relevant for unification.
        This method extracts the last world from each prefix.

        Args:
            lit_1: First literal.
            lit_2: Second literal.

        Returns:
            Tuple of prepared prefixes containing only the last world.
        """
        if lit_2.prefix is None:
            lit_2.prefix = Function("string")
        if lit_1.prefix is None:
            lit_1.prefix = Function("string")
        pre_1 = Function("string", args=lit_1.prefix.args[-1:])
        pre_2 = Function("string", args=lit_2.prefix.args[-1:])
        return pre_1, pre_2

    def _admissible_pairs(self) -> List[Tuple[Function, Function]]:
        """Generate admissible prefix pairs for varying domains in S5.

        For varying domains, additional prefix unification is required between
        variables and their substitutions. For constant and cumulative domains,
        no additional prefix unification is needed.

        Returns:
            List of prefix pairs that need to be unified for varying domains,
            empty list for constant and cumulative domains.
        """
        if self.settings.domain in ["constant", "cumulative"]:
            return []
        if self.settings.domain == "varying":
            equations = []
            for var, term in self.substitution.to_dict().items():
                for eigen in self._find_eigenvariables(term):
                    pre_1 = Function("string", args=var.prefix.args[-1:])
                    pre_2 = Function("string", args=eigen.prefix.args[-1:])
                    equations.append((pre_1, pre_2))
            return equations
