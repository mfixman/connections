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

Each file in [src/axiom_prediction/models/](src/axiom_prediction/models/)
defines a class with the same TitleCase name. All inherit
`AxiomPredictionNetwork`, the shared `torch.nn.Module` base in
[base.py](src/axiom_prediction/models/base.py).
Select a file/class with `train --network SmallFull` or `--network SmallFull.py`.
The available names appear in both top-level and training `--help`.

| Size | Full inputs | No complement edges | No explicit term structure | Width / rounds / hidden layers |
| --- | --- | --- | --- | --- |
| Small | SmallFull | SmallNoComplements | SmallNoTerms | 32 / 2 / 1 |
| Default | DefaultFull | DefaultNoComplements | DefaultNoTerms | 64 / 3 / 2 |
| Large | LargeFull | LargeNoComplements | LargeNoTerms | 128 / 4 / 3 |

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

`--dataset` overrides `--data-dir/dataset`. Collection writes JSONL shards
directly into that directory; training and evaluation only read them.
Single-category/directory collection uses its name for the shard; other inputs
use `problems.jsonl`. Collection resumes using records in `.cache`.
Finish collection before launching readers, and avoid simultaneous writers
to the same shard. Separate policies still require separate datasets.

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
Without a run name, training writes `DATA_DIR/model/model.pt`; with a name,
it writes `DATA_DIR/models/NAME/model.pt`. Existing checkpoints are protected:
use a new run name/directory or pass `--resume` to continue training.

A saved dataset's collection policy is authoritative. Explicitly requesting
a different `--policy` fails; collect a separate dataset to change labels.
Run and evaluate inherit the checkpoint's label policy; conflicting explicit
policies are rejected.

Splits hash the **problem prefix**, such as `ABC123`, keeping all variants
of that problem in the same part. Non-TPTP filenames use their name without
the directory or extension. Use the same split across training, validation,
and testing; evaluation warns when a checkpoint uses an older split scheme.
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
python axiom_predictor.py run SYN --policy SatCoP

# Guided by the named checkpoint:
python axiom_predictor.py run SYN --policy SatCoP \
  --data-dir artifacts/SATCoP --model-name SmallFull
```

A selected checkpoint uses weighted guidance at temperature 1 with all
axioms retained. Without a checkpoint, search is unguided. The CLI no longer
exposes mode, temperature, or axiom filtering.

See the [complete command-line reference](CLI.md) for every remaining option.
