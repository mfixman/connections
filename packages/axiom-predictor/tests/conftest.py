from pathlib import Path
import pytest


@pytest.fixture
def tiny_problem_path():
    return Path(__file__).parent / "fixtures/problems/tiny_theorem.p"
