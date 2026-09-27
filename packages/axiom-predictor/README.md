# Axiom predictor

Collect source-clause SAT cores, train a graph model to rank axioms, and guide
SATCoP or SATResetCoP with its predictions. The graph and checkpoint formats
are preserved from the learncop axiom_predictor branch.

From the repository root:

```bash
uv sync --all-packages --python 3.12
uv run --package axiom-predictor axiom-predictor collect examples/socrates.p --data-dir artifacts/axioms
uv run --package axiom-predictor axiom-predictor train --data-dir artifacts/axioms
uv run --package axiom-predictor axiom-predictor --help
```

Use `--tptp /path/to/TPTP` when resolving benchmark names or included axioms.
The predictor also supports the existing `TPTP` environment variable for its
command-line interface; the connections library receives explicit paths.
Install W&B with `uv sync --package axiom-predictor --extra tracking`,
or `pip install wandb` in a pip environment.

The model uses its original ReLU graph encoder. Fredrik's imitation package
has a separate tanh action model; changing that model would invalidate
existing predictor checkpoints. Dataset schemas retain their learncop names
to allow existing datasets and checkpoints to be loaded.

For pip, run from the repository root:

```bash
python -m pip install -e '.[sat]' -e packages/axiom-predictor -e packages/pycop
python axiom_predictor.py --help
```

The standalone launcher can also be invoked by absolute path from another
directory. Building Pydical requires Git and a C++ compiler.

Collect and shard by TPTP category, train on selected splits, and evaluate
without repeating proof search:

```bash
axiom-predictor collect SYN --tptp /path/to/TPTP --data-dir artifacts/axioms --num-workers 8
axiom-predictor train --data-dir artifacts/axioms --split 4 --parts 0 1 2 --no-wandb
axiom-predictor evaluate --data-dir artifacts/axioms --split 4 --parts 3 --no-wandb
axiom-predictor run examples/socrates.p --data-dir artifacts/axioms --mode strict --policy satresetcop --no-wandb
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
retain their previous command-line behavior. See `axiom-predictor COMMAND
--help` for the complete argument list.
