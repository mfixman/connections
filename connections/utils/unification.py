from typing import Any, Dict, List, Optional, Tuple, Union

from connections.utils.primitives import Expression, Function, Variable


class Substitution:
    """A substitution class implementing union-find with incremental updates and backtracking.

    This class maintains a substitution mapping from variables to terms, with support for:
    - Finding the representative of a term
    - Unifying terms
    - Checking for variable occurrence
    - Backtracking changes
    - Incremental updates

    Attributes:
        parent: Dictionary mapping terms to their representatives.
        trail: List of changes for backtracking.
    """

    def __init__(self) -> None:
        """Initialize an empty substitution."""
        self.parent: Dict[Any, Any] = {}
        self.trail: List[List[Any]] = []

    def find(self, item: Any, add: bool = True) -> Any:
        """Find the representative of a term.

        Args:
            item: The term to find the representative of.
            add: Whether to add the term if not found.

        Returns:
            The representative of the term.
        """
        if not isinstance(item, Variable):
            return item
        if item not in self.parent:
            if not add:
                return item
            # Ensure we have a place to track changes
            if not self.trail:
                self.trail.append([])
            self.trail[-1].append(item)
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            old = self.parent[item]
            self.parent[item] = self.find(self.parent[item], add)
            if old != self.parent[item]:
                # Ensure we have a place to track changes
                if not self.trail:
                    self.trail.append([])
                self.trail[-1].append((item, old, self.parent[item]))
        return self.parent[item]

    def union(self, s: Any, t: Any) -> bool:
        """Attempt to unify two terms.

        Args:
            s: First term to unify.
            t: Second term to unify.

        Returns:
            True if unification is successful, False otherwise.
        """
        self.trail.append([])
        equations = [(s, t)]

        while equations:
            s, t = equations.pop()
            s = self.find(s)
            t = self.find(t)

            if s == t:
                continue

            if isinstance(s, Variable):
                if self.occurs_check(s, t):
                    return False
                self.trail[-1].append((s, self.parent[s], t))
                self.parent[s] = t
            elif isinstance(t, Variable):
                if self.occurs_check(t, s):
                    return False
                self.trail[-1].append((t, self.parent[t], s))
                self.parent[t] = s
            else:
                if s.symbol != t.symbol or len(s.args) != len(t.args):
                    return False
                for arg1, arg2 in zip(s.args, t.args):
                    equations.append((arg1, arg2))
        return True

    def occurs_check(self, var: Variable, term: Any) -> bool:
        """Check if a variable occurs in a term.

        Args:
            var: The variable to check for.
            term: The term to check in.

        Returns:
            True if the variable occurs in the term, False otherwise.
        """
        term_root = self.find(term, add=False)
        if var == term_root:
            return True
        if isinstance(term_root, Expression):
            return any(self.occurs_check(var, arg) for arg in term_root.args)
        return False

    def backtrack(self) -> None:
        """Undo the most recent set of changes."""
        updates = self.trail.pop()
        for action in reversed(updates):
            if isinstance(action, Variable):
                var = action
                del self.parent[var]
                continue
            var, old_state, _ = action
            self.parent[var] = old_state

    def update(self, update: List[Any]) -> None:
        """Apply a set of updates to the substitution.

        Args:
            update: List of updates to apply.
        """
        self.trail.append(update)
        for action in update:
            if isinstance(action, Variable):
                var = action
                self.parent[var] = var
                continue
            var, _, new_state = action
            self.parent[var] = new_state

    def can_unify(self, s: Any, t: Any) -> Tuple[bool, List[Any]]:
        """Check if two terms can be unified.

        Args:
            s: First term to check.
            t: Second term to check.

        Returns:
            Tuple of (success, updates) where success is True if unification is possible.
        """
        unify, updates = self.unify(s, t)
        self.backtrack()
        return unify, updates

    def unify(self, s: Any, t: Any) -> Tuple[bool, List[Any]]:
        """Attempt to unify two terms.

        Args:
            s: First term to unify.
            t: Second term to unify.

        Returns:
            Tuple of (success, updates) where success is True if unification is successful.
        """
        unify = self.union(s, t)
        updates = self.trail[-1]
        return unify, updates

    def equal(self, s: Any, t: Any) -> bool:
        """Check if two terms are equal under the current substitution.

        Args:
            s: First term to compare.
            t: Second term to compare.

        Returns:
            True if the terms are equal, False otherwise.
        """
        s = self.find(s, add=False)
        t = self.find(t, add=False)
        if s == t:
            return True
        if isinstance(s, Expression) and isinstance(t, Expression):
            return (
                s.symbol == t.symbol
                and len(s.args) == len(t.args)
                and all(self.equal(arg1, arg2) for arg1, arg2 in zip(s.args, t.args))
            )
        return False

    def __call__(self, term: Any) -> Any:
        """Apply the substitution to a term.

        Args:
            term: The term to apply the substitution to.

        Returns:
            The term with all variables replaced by their values.
        """
        term_root = self.find(term, add=False)
        if isinstance(term_root, Variable):
            return term_root
        return type(term_root)(
            term_root.symbol, [self(arg) for arg in term_root.args], term_root.prefix
        )

    def to_dict(self) -> Dict[Variable, Any]:
        """Convert the substitution to a dictionary.

        Returns:
            Dictionary mapping variables to their values.
        """
        substitutions = {}
        for var in self.parent:
            if isinstance(var, Variable) and var in self.parent:
                term = self.find(var, add=False)
                if term != var:
                    substitutions[var] = term
        return substitutions

    def __repr__(self) -> str:
        """Return a string representation of the substitution.

        Returns:
            A string representation of the substitution dictionary.
        """
        return repr(self.to_dict())


if __name__ == "__main__":
    sub = Substitution()
    s = Function(
        "f",
        [
            Function("h", [Variable("x1"), Variable("x2"), Variable("x3")]),
            Function("h", [Variable("x6"), Variable("x7"), Variable("x8")]),
            Variable("x3"),
            Variable("x6"),
        ],
    )
    t = Function(
        "f",
        [
            Function(
                "h",
                [
                    Function("g", [Variable("x4"), Variable("x5")]),
                    Variable("x1"),
                    Variable("x2"),
                ],
            ),
            Function("h", [Variable("x7"), Variable("x8"), Variable("x6")]),
            Function("g", [Variable("x5"), Constant("a")]),
            Variable("x5"),
        ],
    )
    unify, updates = sub.can_unify(s, t)
    print(updates)
    print(sub.parent)
    sub.update(updates)
    print(sub.parent)

#### Robinson
#### Too many copies during substitution
# def subst(theta, term):
#     """Applies a substitution theta to a term (Variable or Expression)."""
#     if isinstance(term, Variable):
#         if term in theta:
#             return subst(theta, theta[term])
#         else:
#             return term
#     substituted_args = [subst(theta, arg) for arg in term.args]
#     return type(term)(term.symbol,
#                       substituted_args,
#                       term.prefix)

# def occurs_in(x, y):
#     if isinstance(y, Variable):
#         return x == y
#     elif isinstance(y, Expression):
#         return any(occurs_in(x, arg) for arg in y.args)
#     return False

# def unify_var(var, x, theta):
#     if var in theta:
#         return unify(theta[var], x, theta)
#     elif isinstance(x, Variable) and x in theta:
#         return unify(var, theta[x], theta)
#     elif occurs_in(var, x):
#         return None  # Occur check fails
#     else:
#         theta[var] = x
#         return apply_substitution(theta)

# def unify(x, y, theta=None):
#     if theta is None:
#         theta = {}

#     if x == y:
#         return theta
#     elif isinstance(x, Variable):
#         return unify_var(x, y, theta)
#     elif isinstance(y, Variable):
#         return unify_var(y, x, theta)
#     elif isinstance(x, Expression) and isinstance(y, Expression):
#         if x.symbol != y.symbol or len(x.args) != len(y.args):
#             return None
#         for arg1, arg2 in zip(x.args, y.args):
#             theta = unify(arg1, arg2, theta)
#             if theta is None:
#                 return None
#         return apply_substitution(theta)
#     else:
#         return None

# def apply_substitution(theta):
#     """Applies substitution to itself to ensure idempotence."""
#     for var in list(theta.keys()):
#         theta[var] = subst(theta, theta[var])
#     return theta
