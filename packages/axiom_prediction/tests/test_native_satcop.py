import os
import sys
import time

import pytest

from axiom_prediction.satcop import run_process
from axiom_prediction.satcop_dump import axiom_clause_ids, parse_matrix, parse_proof

# `satcop tiny.p --collect --clausify` for p(a), p(X) => q(X) |- q(a).
clausified_stdout = '''% SATCoP terms Block([3, 2, 0, 0, 2, 3, 4, 3, 4, 0])
% SATCoP literals Block([#1, ¬#4, #6, ¬#8])
% SATCoP symbol #0 0
% SATCoP symbol #1 2
% SATCoP symbol #2 1
% SATCoP symbol #3 0
% SATCoP symbol #4 1
% SATCoP clause #0 #0 #0-#1 false
% SATCoP clause #1 #1 #1-#3 false
% SATCoP clause #2 #0 #3-#4 true
cnf(c0, axiom, p(a)). % 
cnf(c1, axiom, ~p(X0) | q(X0)). % 
cnf(c2, negated_conjecture, ~q(a)). % 
'''

proof_stdout = "\n".join([
    *clausified_stdout.splitlines()[:10],
    "% SZS status Theorem for tiny",
    "% SZS output start ListOfCNF for tiny",
    "% SATCoP core 2",
    "% SATCoP core 0",
    "% SATCoP core 0",
    "% SZS output end ListOfCNF for tiny",
])

def test_matrix_dump_before_search_matches_the_proof_dump():
    before = parse_matrix(clausified_stdout)
    after, core = parse_proof(proof_stdout)
    assert list(map(str, before.clauses)) == list(map(str, after.clauses))
    assert before.conjecture_clauses == after.conjecture_clauses == (2,)
    assert axiom_clause_ids(after) == [0, 1]
    assert core == [0, 2]

    # The variable of clause 1 is shared by its literals, and symbol IDs name the symbols.
    first, second = after.clauses[1].literals
    assert first.atom.args == second.atom.args
    assert first.atom.symbol != second.atom.symbol

def test_truncated_or_changed_dumps_are_rejected():
    with pytest.raises(ValueError, match = "complete successful proof"):
        parse_proof(proof_stdout.replace("% SZS output end ListOfCNF", "% truncated"))

    with pytest.raises(ValueError, match = "literal format"):
        parse_proof(proof_stdout.replace("Block([#1, ¬#4, #6, ¬#8])", "Block([broken])"))

    with pytest.raises(ValueError, match = "out-of-matrix"):
        parse_proof(proof_stdout.replace("core 2", "core 9"))

    with pytest.raises(ValueError, match = "unsupported SATCoP dump record"):
        parse_proof(proof_stdout.replace("symbol #1 2", "symbol #7 2"))

def test_native_wrapper_reaps_a_panicking_search_process(tmp_path):
    log_path = tmp_path / "panic.log"
    code = 'import time; print("thread satcop panicked at failure", flush=True); time.sleep(30)'
    started = time.monotonic()
    with log_path.open("w") as log_file:
        with pytest.raises(RuntimeError, match = "search panicked"):
            run_process([sys.executable, "-c", code], dict(os.environ), log_file, log_path, 10)

    assert time.monotonic() - started < 3
