from typing import Any, List, Optional, Tuple

import connections.utils.unification_d as d
from connections.calculi.intuitionistic import IConnectionState
from connections.utils.primitives import Function


class DConnectionState(IConnectionState):
    """State representation for D modal connection calculus.

    Extends the intuitionistic connection calculus with D-specific prefix handling.
    The D modal logic requires serial accessibility relations between worlds.

    This implementation supports constant, cumulative, and varying domains.
    For cumulative domains, prefix unification is required between variables
    and their substitutions up to the length of the eigenvariable's prefix.
    For varying domains, full prefix unification is required.

    Unlike other modal logics, D does not append fresh variables to prefixes
    during unification.
    """

    def pre_unify(self, pre_1: List[Any], pre_2: List[Any], s: Any) -> Optional[Any]:
        """Attempt to unify two prefixes in D modal logic.

        Args:
            pre_1: First prefix to unify.
            pre_2: Second prefix to unify.
            s: Current substitution.

        Returns:
            The unified substitution if successful, None otherwise.
        """
        return d.pre_unify(pre_1.args, [], pre_2.args, s=s)

    def pre_unify_list(
        self, prefix_list: List[Tuple[Any, Any]], s: Any
    ) -> Optional[Any]:
        """Attempt to unify a list of prefix pairs in D modal logic.

        Args:
            prefix_list: List of prefix pairs to unify.
            s: Current substitution.

        Returns:
            The unified substitution if successful, None otherwise.
        """
        return d.pre_unify_list(prefix_list, s)

    def _pre_eq(self, lit_1: Any, lit_2: Any) -> Tuple[Function, Function]:
        """Prepare prefixes for unification between two literals in D.

        In D, the entire prefix is relevant for unification, but unlike other
        modal logics, no fresh variables are appended to the prefixes.

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
        """Generate admissible prefix pairs based on domain type in D.

        The prefix unification requirements vary by domain type:
        - Constant domains: No prefix unification needed
        - Cumulative domains: Unify prefixes up to eigenvariable length
        - Varying domains: Full prefix unification required

        Returns:
            List of prefix pairs that need to be unified.
        """
        if self.settings.domain in "constant":
            return []
        if self.settings.domain == "cumulative":
            equations = []
            for var, term in self.substitution.to_dict().items():
                for eigen in self._find_eigenvariables(term):
                    var_pre = Function(
                        "string", var.prefix.args[: len(eigen.prefix.args)]
                    )
                    equations.append((var_pre, eigen.prefix))
            return equations
        if self.settings.domain == "varying":
            equations = []
            for var, term in self.substitution.to_dict().items():
                for eigen in self._find_eigenvariables(term):
                    equations.append((var.prefix, eigen.prefix))
            return equations
