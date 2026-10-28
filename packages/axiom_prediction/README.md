# Axiom predictor

Collect SAT-core labels, train graph networks to rank axiom clauses, and use
their predictions to guide SATCoP or SATResetCoP.

From the repository root, with Python 3.12 or newer:

```bash
python -m pip install -r requirements.txt
python axiom_predictor.py --help
python axiom_predictor.py train --help
```

Training defaults: seed 0, 200 epochs, batch size 43, CUDA required, automatic
W&B tracking with warning-only fallback, and maximum available CPUs within
the job allocation. Use `--device auto` to allow CPU fallback, or
`--device cpu` to require CPU execution. Building Pydical requires Git and
a C++ compiler. Install optional tracking with `python -m pip install wandb`.

## Models

Each file in [models/](models/)
defines a class with the same TitleCase name. All inherit
`AxiomPredictionNetwork`, the shared `torch.nn.Module` base in
[base.py](models/base.py).
Select a file/class with `train --network SmallFull` or `--network SmallFull.py`.
The available names appear in both top-level and training `--help`.

| Size | Full inputs | No complement edges | No explicit term structure | Width / rounds / hidden layers |
| --- | --- | --- | --- | --- |
| Small | SmallFull | SmallNoComplements | SmallNoTerms | 32 / 2 / 1 |
| Default | DefaultFull | DefaultNoComplements | DefaultNoTerms | 64 / 3 / 2 |
| Large | LargeFull | LargeNoComplements | LargeNoTerms | 128 / 4 / 3 |
| Extra large | ExtraLargeFull | ExtraLargeNoComplements | ExtraLargeNoTerms | 256 / 5 / 4 |

All use ReLU. `DefaultFull` is the original network and the default choice.
The input variants retain axioms, conjectures, and pooled scoring contexts.
NoTerms removes term/variable nodes and their incident relations, but still
receives clause groundness, predicate arity, and complement edges computed
from the original terms. NoComplements removes only complement edges;
shared-symbol and shared-term connections remain.

To add a variant, add a TitleCase Python file/class to this directory. Set
its `default_config`, and override construction or `forward` if needed.
The constructor must accept the saved `AxiomModelConfig`. No CLI registry
edit is needed. For example:

```python
from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class MyNetwork(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(hidden_dim=48, message_rounds=2)
```

Checkpoints record the selected class and its resolved configuration.
Evaluation, prediction, and guided search restore both automatically.
Version-2 checkpoints remain loadable as the original base network; new
checkpoints use version 3 so older readers cannot silently discard the
selected class.

## Experiments

### Sharing one dataset across runs

Collect once, then give each training job its own output directory:

```bash
python axiom_predictor.py collect SYN \
  --dataset datasets/Reset --policy SatResetCoP

python axiom_predictor.py train \
  --dataset datasets/Reset --data-dir runs/Reference --network DefaultFull \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py train \
  --dataset datasets/Reset --data-dir runs/Smaller --network SmallFull \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py evaluate \
  --dataset datasets/Reset --data-dir runs/Reference --split 10 --parts 8
```

`--dataset` overrides `--data-dir/dataset`. Collection writes CSV shards
directly into that directory; training and evaluation only read them.
Single-category/directory collection uses its name for the shard; other inputs
use `problems.csv`. Collection appends resumable results to a single `cache.csv`
inside each shard's `.cache` directory. Old per-problem JSON caches are read
automatically, including partial results from interrupted collections.
Finish collection before launching readers, and avoid simultaneous writers
to the same shard. Separate policies still require separate datasets.

### Collecting with native Rust SATCoP

`SatCoP` selects Rust SATCoP for collection. `SATCoPCon` selects the previous
Python implementation; `SatResetCoP` keeps its existing meaning. Historical
datasets and checkpoints labelled `satcop` without Rust backend provenance
are interpreted as `SATCoPCon` when training or inheriting a checkpoint policy.

Build the [`axiom_selection` branch](https://github.com/mfixman/satcop/tree/axiom_selection)
of `~/TP/satcop` following its README. It needs
PicoSAT compiled with `--trace`: the clause-core API requires that feature.
The unmodified SATCoP executable has no matrix debug dump and is rejected
with a build instruction. There is no automatic build or source modification.

If the system PicoSAT library lacks trace support, build a private copy:

```bash
cd ~/TP/satcop
curl -fL https://fmv.jku.at/picosat/picosat-965.tar.gz -o /tmp/picosat-965.tar.gz
tar -xzf /tmp/picosat-965.tar.gz -C /tmp
(cd /tmp/picosat-965 && ./configure.sh -O --trace && make -j4 libpicosat.a)
cp /tmp/picosat-965/libpicosat.a .
RUSTFLAGS="-L native=$PWD" cargo build --release
```

Use the same `RUSTFLAGS` for later builds. The library is ignored by Git.

From this repository, either launcher accepts:

```bash
python axiom_selector.py collect SYN --policy SatCoP \
  --satcop-location ~/TP/satcop --dataset datasets/NativeSATCoP \
  --timeout-seconds 900 --num-workers 8 --seed 123

python axiom_selector.py train --dataset datasets/NativeSATCoP \
  --data-dir runs/NativeSATCoP --network SmallFull \
  --split 10 --parts 0 1 2 3 4 5 6 7 --evaluate 8
```

`--satcop-location` accepts a checkout containing `target/release/satcop` or
an executable; its default is `~/TP/satcop`. On Linux, Python uses `taskset` to
give each child one CPU, so SATCoP starts one search thread. `--seed 123` is
forwarded unchanged as `satcop ... --seed 123`; that thread receives seed 123.
The original SATCoP has an internal seeding method but needs the small CLI
addition to expose it. Native SATCoP uses the wall-clock budget; `--step-limit` applies only
to Python policies. Collection resumes from the existing per-problem cache,
including failure records, and refuses to combine results from different
executables, seeds or budgets in one cache. Use a separate dataset for each
collection configuration.

Successful native JSON exports and command/output logs are retained under
`DATASET/.cache/SHARD/native/`. SATCoP's `--collect` flag prints its existing
debug representations; its proof output identifies
each ground core clause's matrix origin. Python captures stdout, parses it,
and writes the JSON files. Saved v1 and v2 JSON exports remain readable.
The converter uses SATCoP's own clauses and
clause IDs, including equality axioms and definitions, without reclausifying
the original input. An axiom is positive if at least one of its ground
instances occurs in PicoSAT's observed UNSAT core; this core is not necessarily
minimal. Conjecture clauses are context rather than prediction targets.
Native symbol IDs distinguish generated symbols from equally named input
symbols. Symbol display names are generated from their IDs. Python records the
seed, observed checkout revision, binary hash and matrix hash as graph metadata.
The minimal dump omits source formula names, search constraints and the start set.
`checkout_revision` describes the checkout at collection time;
the binary hash identifies the executable actually run.

Raw stdout or existing JSON exports can also be converted separately:

```bash
taskset -c 0 ~/TP/satcop/target/release/satcop problem.p --collect --seed 123 --time 900 > proof.out
python packages/axiom_prediction/satcop.py proof.out \
  --problem problem.p --seed 123 --timeout-seconds 900 \
  --output datasets/ImportedSATCoP/proofs.csv
```

Replace CPU 0 with a CPU allowed by your allocation. Raw dumps need `--problem`
because Rust no longer exports the input path; JSON exports already contain it.

The debug parser matches this SATCoP revision and fails on changed layouts or
incomplete proofs. Matrix decoding, JSON serialization, process monitoring,
and training graph construction live in the axiom predictor.

The converter accepts multiple exports for the same problem. Training retains
one example per distinct positive axiom-clause set for each problem, so repeated
cores have no extra weight. Shards may use different search seeds; collect each
seed in a separate cache and give its CSV shard a unique name in the training
dataset directory. Policy, backend and label semantics must match, and repeated
problems must have matching graphs and clause identities. Predictor and prover
revision differences are retained as provenance rather than compatibility checks.
`run --policy SatCoP` searches with Rust SATCoP; see [Native SATCoP search](#native-satcop-search).
Python guided search uses `--policy SATCoPCon`; using a native-trained
model there is transfer across different clausifications.

The Python-written `satcop.collection.v3` matrix stores symbols as `[name, arity,
predicate]` and literals as `[positive, atom_term_id]`. Clause rows contain
`[variable_count, literal_start, literal_stop, is_goal]`;
ranges exclude their stop index. A term starts with a positive symbol ID or
a nonpositive variable code (`-variable_id`). A symbol's arity determines how
many following cells contain argument term IDs. The converter expands this representation
before using the existing graph builder.

For the initial four-run comparison, use DefaultFull, SmallFull,
DefaultNoTerms and DefaultNoComplements. Keep the dataset and policy fixed,
train on parts 0–7, validate on part 8, and reserve part 9 for final proof
search against an unguided baseline with the same policy and budgets.
LargeFull is an optional fifth run. See [CLI.md](CLI.md) for the commands.

### Keeping the dataset inside a run directory

Collect separately for each label policy; networks can reuse each dataset:

```bash
python axiom_predictor.py collect SYN \
  --data-dir artifacts/SATCoP --policy SatCoP

python axiom_predictor.py train \
  --data-dir artifacts/SATCoP --network SmallFull --model-name SmallFull \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py evaluate \
  --data-dir artifacts/SATCoP --model-name SmallFull --split 10 --parts 8
```

Repeat with any network name from the table. `--network` selects the Python
network implementation for training; `--model` selects a trained checkpoint
for proof search, and `--model-name` names the saved run.
Without a run name, training writes `DATA_DIR/model/model.ts`; with a name,
it writes `DATA_DIR/models/NAME/model.ts`.

New files are TorchScript archives that Vampire can load directly, with the original
model configuration, weights, optimizer and RNG state embedded for Python loading
and exact training resumes. Connections accepts both `.ts` and legacy `.pt` files;
a directory selects `model.ts` first, then `model.pt`. Only `.ts` files are written.
Old Vampire-only exports without Connections metadata cannot be loaded by Connections.
Existing checkpoints are protected:
use a new run name/directory or pass `--resume` to continue training.

A saved dataset's collection policy is authoritative. Explicitly requesting
a different `--policy` fails; collect a separate dataset to change labels.
Run and evaluate inherit the checkpoint's label policy; conflicting explicit
policies are rejected.

Choose `--split-scheme 3`, `4`, or `5` on any command (default: **5**).

| Scheme | Problem key | Hash |
| --- | --- | --- |
| 3 | TPTP prefix (e.g. `ABC123`); otherwise filename stem | BLAKE2b of `N\0key`, 8-byte digest, personalization `lcsplit` |
| 4 | As 3, but Mizar names use the article before `__` | Same as 3 |
| 5 | Same as 4 | MD5 of the key only |

All schemes interpret the full digest as a big-endian integer modulo `--split N`.
Schemes 3 and 4 reproduce their historical assignments; changing N rehashes the
inputs, so subdividing a part does not preserve membership. Use the same scheme
and partition count for training and held-out evaluation.

Split scheme 5 hashes only the **problem key** with MD5, interprets the full
digest as a big-endian integer, and takes its remainder modulo `--split N`.
The key is the TPTP prefix (e.g. `ABC123`), the Mizar article name before
`__`, or otherwise the filename without its directory or extension.
Thus `--split 20 --parts 9 19` selects exactly the same problems as
`--split 10 --parts 9`; variants of a problem stay together.
This changes assignments from older split schemes. Evaluation warns about
checkpoints with different or unknown schemes, and existing output files cannot resume across schemes.
The trainer saves a checkpoint after every epoch, independently of logging,
including optimizer/RNG state. It has no validation-based early stopping.
For example, repeat the same training command with `--resume --epochs 200`
to continue to epoch 200, not to train 200 additional epochs. The selected
data, network, seed, split, and optimizer configuration must match. Older
checkpoints remain usable for prediction but cannot resume without this state.

TPTP discovery uses `--tptp`, then `$TPTP`, then conventional local corpus
directories. Here it discovers the sibling `../TPTP`. No corpus is downloaded
automatically. See the [argument reference](CLI.md) for the full search order.

## Proof search

```bash
# Unguided baseline:
python axiom_predictor.py run SYN --policy SATCoPCon

# Guided by the named checkpoint:
python axiom_predictor.py run SYN --policy SATCoPCon \
  --data-dir artifacts/SATCoP --model-name SmallFull
```

A selected checkpoint uses weighted guidance at temperature 1 with all
axioms retained. Without a checkpoint, search is unguided. The CLI no longer
exposes mode, temperature, or axiom filtering.

See the [complete command-line reference](CLI.md) for every remaining option.

### Native SATCoP search

```bash
python axiom_predictor.py run SYN --policy SatCoP --model torok
```

`--policy SatCoP` (also spelled `SATCoP`) runs `~/TP/satcop/target/release/satcop`
on one CPU, as in collection; build it with the traced PicoSAT above. A bare
`--model NAME` refers to `models/NAME.pt` in this checkout. Without a model it
runs ordinary SATCoP. With a model or `--sine`, Python first runs
`satcop --collect --clausify` to obtain SATCoP's own clausification, scores its
axiom clauses, and passes one weight per clause through `--weights`. SATCoP then
orders each extension's alternative clauses by a weighted random permutation,
proportional to the weight, instead of its uniform shuffle; the seed fixes the
permutation. Conjectures weigh 1, weights are floored at `1e-6`, and the
distribution matches Python Weighted guidance at temperature 1. Start clauses
and literal order are unchanged. Strict mode is unsupported. The clausification,
prediction and search share the `--timeout-seconds` deadline; `--step-limit` is
ignored. Errors keep their native logs in a `satcop-run-*` temporary directory.

Checkpoints trained on Python matrices, such as `torok`, score a different
clausification here: SATCoP's definitions, Skolem symbols and equality axioms.

SInE provides a checkpoint-free comparison through `run --sine` and
`evaluate --sine`. It selects axiom clauses using basic SInE (tolerance 1,
unlimited depth, generality threshold 0). Search retains all clauses with the
existing `1e-6` positive weight floor; evaluation scores exact binary decisions
against SAT-core labels, including saved graph datasets. BCE interprets these
scores as probabilities, and top-k ties use clause order. See [CLI.md](CLI.md#sine-comparison)
for usage and resume behavior.
