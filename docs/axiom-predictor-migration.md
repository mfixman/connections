# Consolidation onto connections/main

Base: `83916ba` (27 September 2026). Sources: learncop `aa90df0`, its
connections submodule `4f03edf`, and standalone connections/mfixman `81808a0`.
The port is on `axiom_predictor`; the source branches remain intact.

## What Fredrik changed

This was a consolidation and API rewrite, rather than a directory merge that
could be followed by cherry-picking the learning commits unchanged.

- The intermediate corpus/executor extraction was reorganized in `0eead8a`
  (the history also contains a replayed `99e053f`). Library primitives live
  in `src/connections`; applications are uv workspace packages.
- `952890b` split calculus and run and moved pycop into `packages/pycop`.
  The former `provers.pycop` import namespace is now `pycop`.
- `09d574f` replaced `Prover` with run functions. Later changes renamed the
  layers to `environment` and `interaction`, and the file specification to
  `Problem`. The current entry point is `interaction.run.run_schedule`.
- Policy became `Agent`: a call returns an action or `None`, while
  `AgentStatus` reports why search ended. `Strategy`, `PolicyOptions` and
  `StrategySchedule` still describe construction and budgets. Start-clause
  selection belongs to the agent; conjecture marking belongs to `MatrixOptions`.
- `ee0d652` removed `ProverOutcome`. Runs now return SZS statuses, agent status
  and truncation. A run checks budgets between steps; applications own hard
  process limits. `e09c9e8` and `e0da92e` simplified DFS and iterative deepening.
- `b55a518` explicitly removed SATResetCoP from the refactored design.
- `55c0289` supplied the rewritten imitation/DAgger package: graph schema,
  actor-learner, critic, replay, batching, trainer and experiment CLI.
  It is not an import-compatible copy of learncop's old top-level packages.

## Where the local work went

| Source work | New location |
| --- | --- |
| SATCoP, SATResetCoP, selector-based native SAT cores | `src/connections/agent/sat/` |
| Axiom collection, graph model, checkpoints, splits, W&B, guided runs | `packages/axiom-predictor/` |
| Finite-model search, validation, rendering, model-policy selection | `src/connections/model_finding/` |
| Corpus CLI, policy aliases, resumable comparisons, model CLI | `packages/pycop/src/pycop/` |
| Standalone Python launchers | `axiom_predictor.py`, `run_pycop.py` |

The SAT agents use the new agent interface and `AgentOptions`, and are
constructed by ordinary `PolicyOptions`. Their ground shadow, randomized
sibling search, model ranking, depth gate, resets, core selectors and guided
start/extension ordering are retained. A tableau closed using model lemmas
is never a proof: only UNSAT of the accumulated ground instances sets CLOSED.
Reusing an agent on a fresh state resets its per-problem SAT memory.

The environment changes are limited to all-clause start selection and the
model-lemma rule/record, plus the parser markers needed by model preprocessing.
The scoped recursion guard handles large axiom conjunctions from the old fork.
The imitation graph receives the one-line empty-clause embedding-index fix.
Its architecture and ordinary leanCoP search remain as upstream.

The predictor keeps a private copy of its graph representation and ReLU
encoder. Reusing imitation's rewritten tanh model would change trained model
behavior. Existing `learncop.*` dataset and checkpoint schema names are
intentionally retained. The comparison schema advances to v2: `steps` counts
all actions under the new rollout contract, and `proof_size` is the size of
the current derivation (not the SAT-core size). Failed budgets from the library
are ResourceOut; the comparison supervisor reports Timeout when it kills a worker.
Code fingerprints cover both workspace source roots, so changed prover code
cannot silently reuse comparison results.

The predictor retains the repeating POSIX alarm around collection, now in the
application rather than the core run function. It covers parsing and search;
on non-POSIX systems or non-main threads only cooperative run budgets apply.
The comparison runner supervises worker processes, including parsing.

Four shallow predictor tests were omitted: the `--part` spelling alias,
argparse's required data-directory flag, nonpositive worker parser rejection,
and the hardcoded W&B name prefix. Integration coverage remains for dataset
collection, worker spawning, training, checkpoint loading and W&B behavior.
SAT constructor-default assertions were replaced by behavioral checks for
depth limits, resets, soundness, source cores and episode reuse.

## Running and validation

The complete workspace requires Python 3.12 or newer; the core library still
declares Python 3.10 support. No connections submodule or sibling checkout is
used. Pydical is an optional `connections[sat]` dependency pinned to the same
commit as the old standalone predictor requirements.

```bash
uv sync --locked --all-packages --group docs
uv run --all-packages ruff check .
uv run --all-packages ty check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --all-packages pytest tests packages/pycop/tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --all-packages pytest packages/imitation/tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --all-packages pytest packages/axiom-predictor/tests
uv build --all-packages
uv run --group docs mkdocs build --strict -f docs/mkdocs.yml
```

Package tests run separately because the upstream imitation tests import their
own `conftest` directly. CI runs all three suites with this layout.

Validation on Python 3.13: 302 core/pycop tests passed (4 skipped), 58 imitation
tests passed, and 80 predictor tests passed (1 CUDA test skipped). Ruff and ty
passed, all four packages built as wheels and source distributions, and the
strict documentation build passed. A checkpoint trained by the original
learncop checkout loaded in the migrated package and produced exactly the
same probability, `0.4970654547214508`, on the same tiny fixture. These checks
do not establish full-corpus performance parity or GPU compatibility.

Untracked datasets, presentations, merge-backup files, secrets and local agent
notes in the original checkouts are not imported into Git. No remote is pushed.
The existing `src/connections/Axioms` benchmark symlink is explicitly excluded
from distributions, so building in the original checkout does not package the
local TPTP corpus. The symlink itself is preserved.
