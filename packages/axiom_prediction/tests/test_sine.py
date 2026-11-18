import pytest

from axiom_prediction import sine

@pytest.mark.parametrize("conjecture, expected", [
    ({"common"}, [0, 0, 0, 0, 0]),
    ({"rare"}, [1, 1, 1, 0, 0]),
    ({"other"}, [0, 0, 0, 1, 0]),
    (set(), [0, 0, 0, 0, 0]),
])
def test_fixed_point_and_rare_triggers(conjecture, expected):
    axioms = [
        {"rare", "bridge", "common"}, {"bridge", "end", "common"},
        {"end", "common"}, {"other"}, set(),
    ]
    assert sine.select(axioms, conjecture) == expected

def test_frequency_ties():
    axioms = [{"a", "b"}, {"a", "c"}, {"b", "d"}]
    assert sine.select(axioms, {"a"}) == [1, 0, 0]
    assert sine.select(axioms, {"b"}) == [1, 0, 0]
    assert sine.select(axioms, {"c"}) == [1, 1, 0]
