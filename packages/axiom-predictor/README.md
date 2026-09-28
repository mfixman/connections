# Axiom predictor

Collect source-clause SAT cores, train a graph model to rank axioms, and guide
SATCoP or SATResetCoP with its predictions. The graph and checkpoint formats
are preserved from the learncop axiom_predictor branch.

With Python 3.12 or newer, from the repository root:

```bash
python -m pip install -r requirements.txt
python axiom_predictor.py collect examples/socrates.p --data-dir artifacts/axioms
python axiom_predictor.py train --data-dir artifacts/axioms
python axiom_predictor.py --help
```

Use `--tptp /path/to/TPTP` when resolving benchmark names or included axioms.
The predictor also supports the existing `TPTP` environment variable for its
command-line interface; the connections library receives explicit paths.
Install optional W&B tracking with `python -m pip install wandb`.

The model uses its original ReLU graph encoder. Fredrik's imitation package
has a separate tanh action model; changing that model would invalidate
existing predictor checkpoints. Dataset schemas retain their learncop names
to allow existing datasets and checkpoints to be loaded.

The standalone launcher can also be invoked by absolute path from another
directory. Building Pydical requires Git and a C++ compiler.

Collect and shard by TPTP category, train on selected splits, and evaluate
without repeating proof search:

```bash
python axiom_predictor.py collect SYN --tptp /path/to/TPTP --data-dir artifacts/axioms --num-workers 8
python axiom_predictor.py train --data-dir artifacts/axioms --split 4 --parts 0 1 2 --no-wandb
python axiom_predictor.py evaluate --data-dir artifacts/axioms --split 4 --parts 3 --no-wandb
python axiom_predictor.py run examples/socrates.p --data-dir artifacts/axioms --mode strict --policy satresetcop --no-wandb
```

Use `--sat-policy satcop` during collection to select SATCoP. Core membership
is determined by CaDiCaL failed assumptions; cores need not be minimal and
may vary with solver versions or policies. Labels and predictions refer to
clausified source clauses, including generated equality axioms, rather than
original named TPTP formulas. Conjecture marking is identical during
collection and inference. `--top-k` restricts both the search actions and
the SAT shadow to the selected axioms plus conjecture clauses.

`--model-name NAME` keeps several models beside the same dataset. Dataset
shards, partial worker results, W&B configuration and SLURM CPU detection
retain their previous command-line behavior. See `python axiom_predictor.py COMMAND
--help` for the complete argument list.

## Comparing policies and networks

`--sat-policy satcop` and `--sat-policy satresetcop` select the prover that
collects the SAT-core labels for `collect`, fresh-problem `train`, and
fresh-problem `evaluate`. They do not change the optimizer. Keep a separate
data directory for each collection policy. When training or evaluating a
saved dataset, the policy defaults to its recorded provenance; an explicitly
different policy is rejected rather than silently reusing the old labels.
The `run --policy` option independently selects the policy guided by a model.

Network size is selected with `train --model-preset`:

| Preset | Hidden width | Message rounds | Hidden layers |
| --- | ---: | ---: | ---: |
| `small` | 32 | 2 | 1 |
| `default` | 64 | 3 | 2 |
| `large` | 128 | 4 | 3 |

All use ReLU. `--hidden-dim`, `--message-rounds`, and `--num-hidden-layers`
override individual values; the hidden-layer count applies to the encoder's
action head and the axiom scoring head. Message passing shares weights across
rounds. `--epochs`, `--learning-rate`, and `--weight-decay` control training.
The original predictor is `--model-preset default --graph-input full`.

Choose input information independently with `--graph-input`:

| Input | Information supplied to the graph encoder |
| --- | --- |
| `full` | Original clause, literal, symbol, term, variable features and all relations. |
| `no-complements` | Full graph with the potential complementary-literal edges removed. Other shared-symbol and shared-term connections remain. |
| `no-terms` | Clause, literal and symbol nodes, clause membership, literal-to-predicate links and complement edges. Term/variable nodes and their incident relations are removed. |

All variants retain axioms, conjectures, and the original pooled scoring
contexts. `no-terms` still receives clause groundness, predicate arity, and
complement edges computed from the full problem; it ablates explicit term
structure, not every signal derived from terms. Input ablations leave the
stored dataset intact, so all models can train on the same examples.

For example, collect with each policy and compare all nine size/input pairs:

```bash
for policy in satcop satresetcop; do
  data="artifacts/axioms-$policy"
  python axiom_predictor.py collect SYN --tptp /path/to/TPTP --data-dir "$data" --sat-policy "$policy"
  for size in small default large; do
    for input in full no-complements no-terms; do
      name="$size-$input"
      python axiom_predictor.py train --data-dir "$data" --model-name "$name" \
        --model-preset "$size" --graph-input "$input" --epochs 200 \
        --split 4 --parts 0 1 2 --no-wandb
      python axiom_predictor.py evaluate --data-dir "$data" --model-name "$name" \
        --split 4 --parts 3 --no-wandb
    done
  done
done
```

Use a new `--model-name` for each experiment: retraining the same name replaces
its checkpoint. Checkpoints store the resolved size and input settings;
`evaluate`, `predict`, and guided `run` reconstruct them automatically.
Existing checkpoints lacking an input setting continue to use `full`.
