# The prefix unification for D is a simple pattern matching, i.e. the standard term unification can be used.
from copy import deepcopy
from typing import Any, List, Optional, Tuple

from connections.utils.unification import *
from connections.utils.unification import Substitution
from connections.utils.unification_intu import flatten_list


# sub returned from unify is the sub to pass on to recursive call
def pre_unify(
    l_pre: List[Any],
    m_pre: List[Any],
    r_pre: List[Any],
    s: Substitution = Substitution(),
    counter: int = 0,
) -> Optional[Substitution]:
    """Attempt to find a prefix unifier for D modal calculus.

    This function implements prefix unification for D modal calculus using simple pattern matching.
    Since D modal calculus has a simpler prefix structure, standard term unification can be used.

    Args:
        l_pre: Left prefix list to unify.
        m_pre: Middle prefix list to unify (unused in D modal calculus).
        r_pre: Right prefix list to unify.
        s: Current substitution.
        counter: Counter for tracking unification attempts.

    Returns:
        A substitution if unification is successful, None otherwise.
    """
    l = flatten_list([s(pre) for pre in l_pre])
    r = flatten_list([s(pre) for pre in r_pre])
    if len(l_pre) != len(r_pre):
        return None
    new_s = deepcopy(s)
    for arg_1, arg_2 in zip(l, r):
        unifies, _ = new_s.unify(arg_1, arg_2)
        if not unifies:
            return None
    return new_s


def pre_unify_all(
    l_pre: List[Any],
    m_pre: List[Any],
    r_pre: List[Any],
    s: Substitution = Substitution(),
    unifiers: List[Substitution] = None,
    counter: int = 0,
) -> Tuple[List[Substitution], int]:
    """Find all possible prefix unifiers for D modal calculus.

    Since D modal calculus uses simple pattern matching, there can be at most one unifier.
    This function returns a list containing the single unifier if found, or an empty list.

    Args:
        l_pre: Left prefix list to unify.
        m_pre: Middle prefix list to unify (unused in D modal calculus).
        r_pre: Right prefix list to unify.
        s: Current substitution.
        unifiers: List to store found unifiers (unused in D modal calculus).
        counter: Counter for tracking unification attempts.

    Returns:
        Tuple of (list of unifiers, counter).
    """
    s = pre_unify(l_pre, m_pre, r_pre, s)
    if s is None:
        return [], counter
    return [s], counter


def pre_unify_list(
    equations: List[Tuple[Any, Any]], s: Substitution = Substitution(), counter: int = 0
) -> Optional[Substitution]:
    """Attempt to unify a list of prefix equations.

    This function takes a list of prefix equation pairs and attempts to find a unifier
    that satisfies all equations. Since D modal calculus uses simple pattern matching,
    it recursively tries to find a single unifier that satisfies all equations.

    Args:
        equations: List of prefix equation pairs to unify.
        s: Current substitution.
        counter: Counter for tracking unification attempts.

    Returns:
        A substitution if unification is successful, None otherwise.
    """
    l1, l2 = equations[0]
    equations = equations[1:]
    unifiers, counter = pre_unify_all(l1.args, [], l2.args, s=s)
    if not equations:
        if not unifiers:
            return None
        else:
            return unifiers[0]

    # Recursively try all possible prefix unifiers
    res = None
    for unifier in unifiers:
        res = pre_unify_list(equations, s=unifier)
        if res is not None:
            return res
    return res
