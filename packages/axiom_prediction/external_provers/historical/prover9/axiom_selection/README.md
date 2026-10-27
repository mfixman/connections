# Torok guidance for Prover9

The strongest current development coverage result is described in [RESUME.md](RESUME.md):
a short Torok top-1 search followed by resuming the saved baseline search.
It proves 106/178 problems versus baseline 101 and preserved top-16 103, with
slower common successes. The earlier rule remains documented in [SUPPORT.md](SUPPORT.md). The sections below document the earlier weighted-queue
adapter, which performed worse than baseline.

This is a small adapter from the user's Connections axiom predictor to Prover9
2017-11A. It retains all input clauses and changes given-clause selection. The
original resolution, paramodulation, simplification, proof ancestry, and resource
limits remain in force. Only a small conditional hook was added to
`provers.src/giv_select.c`; all other integration logic lives here.

## What the model learned

`models/torok.ts` is a byte-for-byte copy of the exported model in
`~/TP/vampire/Shell/Torok/torok.ts`. Its predictions were checked against
`~/TP/connections/models/torok.pt`. The checkpoint is ExtraLargeFull at epoch 45:
256-dimensional embeddings, five message-passing rounds, four hidden layers,
and ReLU. Hashes, configuration, provenance, and the training dataset fingerprint
are in `models/provenance.json`.

Connections collects successful SATResetCoP searches and labels input axiom
clauses by native CaDiCaL failed-assumption SAT-core membership. A negative label
means absent from that observed core, not unusable in every proof. Training uses
binary cross-entropy with logits, Adam at learning rate 0.001, batch size 43,
weight decay zero, and seed zero. The checkpoint records a 200-epoch target;
the snapshot is from epoch 45. The saved selection manifest reports 125
available epochs and selects epoch 45 by validation AUC (0.97217, part 8),
not by Prover9 proof performance. It records collection
limits of 120 seconds and one million steps. These are checkpoint settings,
not a reconstruction of every training job's history.

The graph has symbol, clause, literal, term, and variable nodes. Its seven
relations encode containment, atoms, term/variable arguments with positions,
symbol occurrences, literal predicates, and unifiable complementary literals.
Ground terms are shared; variables and non-ground terms are clause-local.
Symbol spellings are not learned embeddings. Clause features include length,
role, groundness, and conjecture status. The scorer combines an axiom embedding,
mean conjecture and axiom contexts, and candidate/conjecture products and
differences. A sigmoid produces the axiom score.

Training used split scheme **4**, ten parts, training parts **0–7**. The pilot
uses part **9** of that same scheme, grouping TPTP variants by their six-character
family key. The current Connections default is scheme 5; using it here would
not establish separation from this model's training data. Scores are trained
for SATResetCoP cores; transfer to Prover9 is an experiment, not an assumption.

This follows the small cross-prover input-guidance experiment proposed in
Connections' `packages/axiom_prediction/plan.txt`. The related iProver GNN paper
in `~/TP/connections/papers/chvalovsky2023iprover_gnn.txt` discusses internal
clause guidance, but this adapter does not retrain a model on derived clauses.

## Selection rule

Four of every five given-clause picks sample the current passive clauses with
probability proportional to their weights. The fifth uses Prover9's existing
selector. Input axioms get `max(1e-6, sigmoid(logit))`; conjecture clauses get 1.
Derived clauses inherit the geometric mean of available earlier parent weights
from their recorded justifications. Parentless introduced clauses get 1.
Inheritance and the four-to-one mixture are new, fixed heuristics; they were
not learned by the checkpoint.

A Fenwick tree supports insertion, removal, and weighted sampling in logarithmic
time, with a reproducible SplitMix64 generator. Clause-ID-indexed storage also
caches inherited weights. It consumes approximately 24 bytes per allocated ID
slot and is allocated separately from LADR's memory accounting. Default Prover9
weight/SOS limits and its normal redundancy deletion still apply. Torok itself
does not filter premises. Changing the schedule does not guarantee completeness
under the default finite resource limits.

The `Uniform` control uses the same mixture and queue but assigns every input
weight 1, allowing randomization to be distinguished from learned guidance.
A baseline run uses the original Prover9 executable and does not load PyTorch.

## Build and run

The adapter is self-contained with respect to Connections: the required parser,
CNF representation and graph builder are frozen under `torok_vendor/`, and the
TorchScript model is included under `models/`. No Connections checkout or
`CONNECTIONS_ROOT` setting is needed. The bundled source retains its license and
provenance in `torok_vendor/LICENSE` and `torok_vendor/provenance.json`.

On the HPC, copy this Prover9 repository including `axiom_selection/torok_vendor/`
and `axiom_selection/models/torok.ts`. Install the ordinary Python dependencies
and build the native executables on that machine:

```sh
python -m pip install -r axiom_selection/requirements.txt
python axiom_selection/build.py
```

Requirements are Python 3.10+, PyTorch, Lark, GCC, make, and Linux for process
control. TPTP remains an external dataset: by default the batch runner looks in
the sibling `TPTP` directory; use `--tptp /path/to/TPTP` to override it. The
batch warm-up uses `Problems/GRP/GRP001-1.p` from that dataset. The existing
experiment command otherwise stays the same.

The isolated-copy test runs baseline, resumed top-1 and weighted top-16 with a
small fixture dataset, a nonexistent Connections root and external Connections
imports forbidden. It checks scoring, proof outcomes, CSV output and summaries:

```sh
python axiom_selection/tests/standalone.py
python axiom_selection/tests/preprocessing.py
```

```sh
python axiom_selection/build.py
python axiom_selection/run.py ../TPTP/Problems/GRP/GRP001-1.p \
  --mode Torok --seconds 20 --seed 1 --output /tmp/torok-result.json
python axiom_selection/run.py ../TPTP/Problems/GRP/GRP001-1.p \
  --mode Baseline --seconds 20 --output /tmp/baseline-result.json
python axiom_selection/tests/check.py
```

Modes are `Baseline`, `Torok`, and `Uniform`. Output includes the exact native
input, proof/search log, and timing JSON. For a strict whole-process wall limit,
use the benchmark driver or an external timeout: `run.py` subtracts preprocessing
from the search budget, but cannot interrupt every Python/native preprocessing
operation itself. The benchmark kills the entire process group at its deadline.

This first adapter deliberately accepts **CNF only**. The bundled parser resolves includes
and produces one shared clause matrix; both modes receive the same native input,
with clause-index labels connecting model scores to proof ancestry. Equality is
restored from Connections' internal representation to Prover9's native equality.
Unsupported symbol encodings fail explicitly. FOF requires a separately audited,
shared clausification step; the old LADR TPTP translator is not used here.

The enhanced executable is `axiom_selection/build/prover9-torok`. Its guidance
is enabled only by `PROVER9_TOROK_WEIGHTS`, pointing to a text file containing a
clause count followed by probabilities in input-label order. Set
`PROVER9_TOROK_SEED` for its seed. With that variable absent it follows the native
selection path. The wrapper sets or clears these variables automatically.

The initial measured result is documented in [RESULTS.md](RESULTS.md): this
configuration solved fewer problems overall than the native baseline.

## Matched performance pilot

```sh
python axiom_selection/benchmark.py \
  --output axiom_selection/results/my-pilot
```

The driver uses at most 18 single-core workers, within an affinity set of at most
19 logical CPUs. It randomly orders known-unsatisfiable CNF families in the
recorded held-out split, selecting one variant per family. Each completed group
runs Baseline once and Torok/Uniform with seeds 1 and 2, in a randomized order on
the same CPU. Each arm gets 15 seconds including Python startup, parsing, graph
construction, model loading, inference, and proof search. No predictions are
precomputed for free. There is no top-k or problem-size cutoff.

It starts groups for 330 seconds, then finishes them (at most another 75 seconds).
The manifest freezes the sample order, model/source hashes, revisions, seeds,
and budgets. Input failures must be reported separately. Compare each seed to
the same baseline; do not present the union of two seeds as a fair 15-second
competitor. The comparison measures this CPU wrapper, including cold model
startup; it is not a measurement of a persistent inference service or a claim
about all TPTP problems. Uniform guidance has no neural-inference overhead.

## Compact comparison results

`compare.py` now saves one CSV per strategy in each run/part directory, for
example `results/part-0/results-Baseline.csv` and
`results/part-0/results-ResumeSupport1.csv`. An `--output trial.csv` argument
names the run directory `trial/`; `--output trial` has the same effect.
There is one row per problem. With `--repeat`, additional repeats use separate
`results-STRATEGY-repeat2.csv`, `-repeat3.csv`, etc., to retain resume identity
without a replicate column.
Only a run manifest and lock file accompany the CSVs; no per-problem files,
aggregate `results.json`, or session source copies are created.

Columns include:

- `problem`, `declared_status`: TPTP filename (including `.p`) and declared status, preserving
  the header spelling. Status is blank if the source/header is unavailable.
- `result`: `Proved`, `Timeout`, `Error`, or `Unproved`. Unproved does not mean
  satisfiable; these are search outcomes, not inferred TPTP statuses.
- `time_ms`: measured wall time of the bounded arm, including parsing, scoring,
  search and result collection. Worker model warm-up precedes the timed arm.
- `steps`: total given-clause count across search stages, when every stage
  emitted statistics. `generated` and `kept` similarly total stage counts.
  Missing statistics stay blank, including proofs stopped before statistics
  were printed. These are not proof lengths or exact counts for killed searches.
- `part`: zero-based job partition, using the run manifest’s partition scheme
  and split count. Blank for legacy runs without partition metadata. This is
  distinct from the model’s training split.
- `strategy`, `proved`, `clauses`, `score_ms`, `error`, and `score_error`
  support paired analysis.
Run settings, split, hashes and timing protocol remain in the run manifest.
The CSV retains only the listed analysis columns, without embedded JSON or
per-stage search details. Summaries report unavailable stage counts as unknown.


Native inputs, proof/search logs and prediction vectors are temporary and are
removed after statistics extraction. Set `TMPDIR` to node-local scratch on HPC
if desired. Full proof artifacts are not embedded in the CSV; use the individual
problem runners when those are needed. Search strategies and budgets are unchanged.

Rerunning the same command recovers completed rows from CSV, repairs a truncated
last row, and runs missing arms. Rows are saved after each paired problem group;
if the driver is interrupted mid-group, that group's unsaved work is repeated.
The existing summary command accepts the run directory (or its `.csv` run label).

Convert a completed legacy run without rerunning searches:

```sh
python axiom_selection/export_results.py path/to/old-run --output path/to/compact-run
```

The exporter reads aggregate and per-problem checkpoints and extracts statistics
from existing logs. Add `--tptp /new/path/to/TPTP` if the dataset moved, to recover
declared statuses. It leaves the original directory intact and refuses to append
to an existing CSV export. The exported directory contains just the strategy CSVs;
these preserve the analysis columns, but not run metadata, per-stage details,
full logs or predictions.
