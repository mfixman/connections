# SATCoP policy

`SATCoPCon` follows the search in the canonical SATCoP source supplied in
`~/packages/satcop`, corresponding to commit
`c35604b58921fdc815f7fdb80bf7cb62775a6014`. The relevant reference files are
`src/search.rs`, `src/ground.rs`, `src/sat.rs`, and `src/builder.rs`.

The default policy now:

- Seeds and visits conjecture clauses in matrix order, falling back to positive
  clauses if there are no conjecture clauses or every clause is a conjecture.
- Grounds unbound variables with the most frequent conjecture constant, or a
  fresh fallback constant, both when learning clauses and consulting the model.
- Checks regularity for the selected goal, then tries reductions from the oldest
  path literal first. Reductions remain available at the depth limit.
- Allows one extension at depth one, shuffles extension candidates and child
  goals, and uses neither factorization nor model-based ranking of connections.
- Visits every start-clause literal. For extensions, it repeatedly selects a
  remaining child whose grounded literal is true in the current SAT assignment.
  False or unknown children are skipped only after no true child remains.
- Checks active-clause tautology constraints after each successful unification,
  including reflexive positive equalities, before adding ground instances.
  The reflexivity axiom itself is exempt, as in the canonical implementation.
- Learns instances of every active clause after a reduction or extension and
  immediately updates the SAT assignment. Successful subgoals are committed;
  failure of a later sibling retries the enclosing extension, not the successful
  sibling's alternatives.
- Repeats a full pass over start clauses at the same depth if it learned any new
  ground clause, increasing depth only after a pass without new instances.
- Maintains a separate seeded SAT random walk with unit forcing and 10,000
  iterations before CDCL fallback. Only an UNSAT shadow establishes a proof.

The policy produces ordinary tableau actions. `ModelLemma` actions represent
skipped extension children, and undo actions restore the bindings of a failed
extension or completed start attempt. A stack of generators implements the
recursive search without depending on Python's recursion limit.

## Compatibility and remaining differences

SATResetCoP uses the previous `search.py` implementation, preserved verbatim as
`reset_search.py`. Its guidance, shadow solver, default options, and restart
rules are unchanged. The axiom-guidance mixin is independent of either policy,
so guided SATResetCoP also retains its original method resolution behavior.
The shared tableau, constraint solver, clausifier, and parser are unchanged.

This is an algorithmic alignment, not a claim of identical runs to the Rust
executable. Python's RNG differs from Rust's SmallRng. The native fallback is
still CaDiCaL through the existing pinned `pydical` dependency, with source
selectors for training labels, rather than PicoSAT. Native models, forced
assignments, and UNSAT cores can therefore differ. Clauses and their order come
from the existing Connections preprocessing; ties for the most frequent goal
constant use deterministic traversal order. Canonical preprocessing and its
elimination of redundant disequality constraints have not been ported.
Connections FOF matrices also dualize literal signs, whereas CNF inputs retain
their signs. The policy recognizes the variable-reflexivity axiom to determine
the equality sign; without that schema it conservatively omits equality-specific
constraints. Complementary-literal tautology constraints work with either sign
convention. Start selection uses the roles and literal signs of the supplied
matrix; marking conjecture clauses is recommended when loading FOF problems.

`AgentOptions.start` and `initial_depth` remain configurable. The canonical
policy always uses its own cuts, regularity, and depth schedule; DFS-specific
`cut`, `scut`, `comp`, `backtrack`, and `factorization` options do not change that
algorithm. Explicit axiom guidance can reorder starts/extensions and restrict
clauses, as before.

## Validation

`tests/unit/agent/test_satreset_regression.py` compares 144 deterministic traces
with snapshots captured at Connections commit `56c9514`, before this change.
They cover 24 first-order matrices, seeds 0 and 42, and unguided, strict, and
weighted SATResetCoP. Each trace includes up to 100 actions, status, depth,
ground clauses, SAT assignments, and source-core diagnostics. These tests check
search behavior, rather than just final proof counts; they do not compare wall
time.

`test_satcop_canonical.py` checks the canonical search decisions and compares
incremental SAT answers and assignments with exhaustive truth tables. The
canonical executable and this policy also both prove the supplied Dreadbury
Mansion (`PUZ001+1.p`) and tiny theorem examples.

Run the tests with an installed `requirements.txt` environment:

```sh
python -m pytest tests/unit packages/axiom_prediction/tests packages/pycop/tests
```
