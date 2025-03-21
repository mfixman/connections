"""Tests for the unification modules."""

import pytest

from connections.utils.primitives import Function, Constant, Variable
from connections.utils.unification import Substitution

@pytest.fixture
def symbols():
    sym = {
        "X": Variable("X"),
        "Y": Variable("Y"),
        "Z": Variable("Z"),
        "a": Constant("a"),
        "b": Constant("b"),
    }
    sym["fxy"] = Function("f", [sym["X"], sym["Y"]])
    sym["fyx"] = Function("f", [sym["Y"], sym["X"]])
    sym["fax"] = Function("f", [sym["a"], sym["X"]])
    sym["fab"] = Function("f", [sym["a"], sym["b"]])
    sym["fx"] = Function("f", args=[sym["X"]])
    sym["fy"] = Function("f", args=[sym["Y"]])
    sym["fyz"] = Function("f", args=[sym["Y"], sym["Z"]])
    sym["ffx"] = Function("f", args=[Function("f", args=[sym["X"]])])
    sym["fa"] = Function("f", args=[sym["a"]])
    sym["gy"] = Function("g", args=[sym["Y"]])
    sym["ga"] = Function("g", args=[sym["a"]])
    sym["gx"] = Function("g", args=[sym["X"]])
    sym["fgx"] = Function("f", args=[sym["gx"]])
    sym["fgxx"] = Function("f", args=[sym["gx"], sym["X"]])
    sym["fya"] = Function("f", args=[sym["Y"], sym["a"]])
    yield sym

def test_substitution():
    """Test the basic substitution functionality."""
    s = Substitution()

    # Test variable substitution
    x = Variable("x")
    y = Variable("y")
    f = Function("f", [x, y])

    # Test find
    assert s.find(x) == x
    assert s.find(y) == y

    # Test union
    assert s.union(x, y)
    assert s.find(x) == y

    # Test backtrack
    s.backtrack()
    assert s.find(x) == x
    assert s.find(y) == y
    
class TestUnify:
    """Tests for unification functionality."""

    def test_basic_unification(self, symbols):
        """Test unification of constants and variables."""
        sub = Substitution()
        assert sub.can_unify(symbols["a"], symbols["a"]) == (True, [])
        assert sub.can_unify(symbols["a"], symbols["b"]) == (False, [])
        assert sub.can_unify(symbols["X"], symbols["X"]) == (True, [symbols["X"]])
        assert sub.can_unify(symbols["a"], symbols["X"]) == (
            True,
            [symbols["X"], (symbols["X"], symbols["X"], symbols["a"])],
        )
        assert sub.can_unify(symbols["X"], symbols["Y"]) == (
            True,
            [symbols["X"], symbols["Y"], (symbols["X"], symbols["X"], symbols["Y"])],
        )

    def test_function_unification(self, symbols):
        """Test unification of function terms."""
        sub = Substitution()
        assert sub.can_unify(symbols["fax"], symbols["fab"]) == (
            True,
            [symbols["X"], (symbols["X"], symbols["X"], symbols["b"])],
        )
        assert sub.can_unify(symbols["fa"], symbols["ga"]) == (False, [])
        assert sub.can_unify(symbols["fx"], symbols["fy"]) == (
            True,
            [symbols["X"], symbols["Y"], (symbols["X"], symbols["X"], symbols["Y"])],
        )
        assert sub.can_unify(symbols["fx"], symbols["gy"]) == (False, [])
        assert sub.can_unify(symbols["fx"], symbols["fyz"]) == (False, [])

    def test_complex_unification(self, symbols):
        """Test unification of more complex terms."""
        sub = Substitution()
        assert sub.can_unify(symbols["fgx"], symbols["fy"]) == (
            True,
            [symbols["Y"], (symbols["Y"], symbols["Y"], symbols["gx"])],
        )
        assert sub.can_unify(symbols["fgxx"], symbols["fya"]) == (
            True,
            [
                symbols["X"],
                (symbols["X"], symbols["X"], symbols["a"]),
                symbols["Y"],
                (symbols["Y"], symbols["Y"], symbols["gx"]),
            ],
        )
        assert sub.can_unify(symbols["fxy"], symbols["fyx"]) == (
            True,
            [symbols["Y"], symbols["X"], (symbols["Y"], symbols["Y"], symbols["X"])],
        )
        assert sub.can_unify(symbols["fxy"], symbols["fab"]) == (
            True,
            [
                symbols["Y"],
                (symbols["Y"], symbols["Y"], symbols["b"]),
                symbols["X"],
                (symbols["X"], symbols["X"], symbols["a"]),
            ],
        )

    def test_occurs_check(self, symbols):
        """Test occurs check during unification."""
        sub = Substitution()
        assert sub.can_unify(symbols["fx"], symbols["ffx"]) == (
            False,
            [symbols["X"]],
        )

        # Test direct occurs check
        x = Variable("x")
        f = Function("f", [x])
        assert not sub.union(x, f)

    def test_invalid_unification(self):
        """Test unification of incompatible terms."""
        sub = Substitution()
        x = Variable("x")
        y = Variable("y")
        f = Function("f", [x])
        g = Function("g", [y])
        assert not sub.union(f, g)

    def test_incremental_unification(self, symbols):
        """Test incremental unification and substitution chains."""
        sub1 = Substitution()
        sub1.unify(symbols["X"], symbols["Y"])
        sub1.unify(symbols["X"], symbols["a"])
        assert sub1.to_dict() == {
            symbols["X"]: symbols["a"],
            symbols["Y"]: symbols["a"],
        }

        sub2 = Substitution()
        sub2.unify(symbols["a"], symbols["Y"])
        sub2.unify(symbols["X"], symbols["Y"])
        assert sub2.to_dict() == {
            symbols["X"]: symbols["a"],
            symbols["Y"]: symbols["a"],
        }

        sub3 = Substitution()
        sub3.unify(symbols["X"], symbols["a"])
        assert sub3.unify(symbols["b"], symbols["X"]) == (False, [])

    def test_backtracking(self):
        """Test backtracking of substitutions."""
        sub = Substitution()
        x = Variable("x")
        y = Variable("y")
        sub.union(x, y)
        sub.backtrack()
        assert sub.find(x) == x
        assert sub.find(y) == y