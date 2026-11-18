from pathlib import Path

import pytest

from axiom_prediction import cli
from axiom_prediction.split import ProblemSplit, split_key

names = [
    "ABC001+1.p", "ABC001-2.p", "ABC002+1.p",
    "struct_0__t3_struct_0.p", "struct_0__t4_struct_0.p",
    "tiny_theorem.p", "nested.name.p", "épreuve.p",
]

# Fixed assignments: every dataset, checkpoint and result since 2026-10-23 depends on them.
@pytest.mark.parametrize("count, expected", [
    (3, [1, 1, 2, 0, 0, 1, 2, 0]),
    (10, [6, 6, 6, 6, 6, 2, 6, 5]),
    (20, [6, 6, 16, 16, 16, 12, 6, 5]),
])
def test_assignments_never_change(count, expected):
    split = ProblemSplit(count)
    assert list(map(split.part, names)) == expected
    assert [split.part(Path("/different/root") / name) for name in names] == expected

def test_parts_partition_the_problems_and_keep_variants_together():
    problems = [f"article{index}__t{index % 3}.p" for index in range(300)]
    parts = [ProblemSplit(10, [part]).select(problems) for part in range(10)]
    assert sorted(sum(parts, [])) == sorted(problems)
    assert ProblemSplit(10, list(range(10))).select(problems) == problems
    assert split_key("ABC001-2.p") == "ABC001" and split_key("article__t1.p") == "article"
    assert ProblemSplit(10, [2]).to_dict() == {"split": 10, "parts": [2], "scheme": 5}

def test_earlier_schemes_are_refused():
    with pytest.raises(ValueError, match = "only split scheme 5"):
        ProblemSplit(scheme = 4)

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--split-scheme", "4"])

def test_validation_subsample_is_a_finer_split_of_the_held_out_parts():
    problems = [f"article{index}__t1.p" for index in range(2000)]
    held_out = ProblemSplit(10, [8, 9])
    chosen = held_out.subsample(held_out.select(problems), 10)
    assert chosen == ProblemSplit(100, [8, 9]).select(problems)
    assert 0 < len(chosen) < len(held_out.select(problems)) / 5
    assert held_out.subsample(problems, 1) == problems
    assert held_out.subsample(problems, 0) == []
