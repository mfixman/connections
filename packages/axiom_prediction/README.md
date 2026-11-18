# Axiom predictor

Learns which axiom clauses of a first-order problem a proof needs, and uses the prediction to guide provers.

1. `collect` proves TPTP problems with SATCoP or SATResetCoP and labels each axiom clause of the clausified problem
   (the matrix) with whether it is in the SAT core of the proof.
2. `join` merges collect results into one deduplicated dataset.
3. `train` fits a graph network that scores every axiom clause of a matrix against its conjecture.
4. `run` searches for proofs again, ordering the prover's choices by the scores of a checkpoint.

The checkpoints are TorchScript files that Prover9 (`external/Prover9/axiom_selection`) and Vampire
(`external/vampire/Shell/Torok`) load directly.

```bash
python -m pip install -r requirements.txt    # from the repository root, Python 3.12 or newer
python axiom_predictor.py COMMAND --help
```

## Files

| File | What it holds |
| --- | --- |
| `cli.py` | The four commands and their arguments. |
| `tptp.py`, `split.py` | Finding and loading problems; assigning them to the parts of a split. |
| `collection.py`, `problems_file.py` | `collect`: the worker that proves one problem, and the `problems.csv` it appends to. |
| `satcop.py`, `satcop_dump.py`, `satcop_search.py` | Native (Rust) SATCoP: running it, decoding its matrix and core, guided search. |
| `search_workers.py`, `parallel.py`, `limits.py` | Worker processes with a timeout each; worker counts; default budgets. |
| `joined.py`, `dataset.py`, `data.py` | `join` and the joined dataset; settings shared by datasets; one training example. |
| `representation/`, `graph.py` | The matrix graph (clauses, literals, terms, variables, symbols) and its batching. |
| `encoder.py`, `models/`, `configuration.py` | The message-passing encoder, the network variants and their sizes. |
| `training.py`, `training_log.py`, `resume.py` | `train`: the loop, its log lines, and exact resumption. |
| `validation_run.py`, `background_validation.py` | Proof search on held-out problems after each epoch, beside training. |
| `model.py`, `torchscript.py` | Checkpoints, the predictor that scores a matrix, and the TorchScript export. |
| `run.py`, `guided.py`, `sine.py` | `run`: one search per worker; the weighted SAT policies; the SInE control. |
| `output.py`, `output_resume.py` | Result rows as CSV, and the resumable `--output` journal. |
| `wandb_tracking.py`, `metrics.py`, `logs.py`, `io.py` | W&B logging, ranking metrics, log lines, CSV and JSON helpers. |
| `prover_cores/` | Labels from the proofs of unmodified E, iProver, Prover9 and Vampire. |
| `experiments/`, `pivot/` | Scripts of past experiments and monitors; not part of the package. |

## The four commands

A typical sequence, with the split used throughout: 10 parts, 0-7 for training, 8 for validation, 9 for testing.

```bash
python axiom_predictor.py collect --dataset results/SatCoP/seed-3/part-0 --split 10 --part 0 \
  --policy SatCoP --seed 3 --satcop-location ~/satcop --tptp ~/TPTP

python axiom_predictor.py join results/SatCoP/seed-*/part-* results/SatResetCoP/seed-*/part-* \
  --dataset datasets/Joined

python axiom_predictor.py train --dataset datasets/Joined --data-dir runs/First --network ExtraLargeFull \
  --split 10 --parts 0 1 2 3 4 5 6 7 --evaluate 8 --validation-run-ratio 0 --max-example-mb 10

python axiom_predictor.py run --split 10 --parts 8 --policy SatResetCoP --model runs/First/model/epoch-0020.ts \
  --cpu --output results/part8-guided.csv
```

Problems are `.p` files, directories of them, TPTP file names or category codes such as `SYN`; without any, `collect`
and `run` take every FOF and CNF problem of the TPTP root (`--tptp`, then `$TPTP`, then a `TPTP` directory near the
checkout). `--split N --parts P ...` keeps the problems of those parts. Scheme 5 (the default) takes the MD5 of the
problem key modulo N, where the key is the TPTP prefix (`SYN012` for `SYN012-3.p`) or the Mizar article before `__`;
variants of one problem therefore share a part, and parts P of N are the union of parts P and P + N of 2N. Schemes 3
and 4 reproduce the assignments of older runs.

### collect

Appends one row per problem to `DATASET/problems.csv` as soon as the problem finishes: the matrix graph and labels
of a proof, or the reason for failing. Running the same command again skips every problem already there, and refuses
to continue with a different policy, seed, timeout or SATCoP executable. Use one directory per policy, seed and part,
and never two writers on one file. Problems that TPTP declares `Satisfiable` or `CounterSatisfiable` are skipped.
`--problems-from FILE` adds the problems listed in a file, for example those that earlier seeds proved.

`--policy SatCoP` runs native SATCoP (below); `SatResetCoP` and `SATCoPCon` are the Python policies of the Connections
prover, whose labels come from CaDiCaL's failed assumptions. The SAT core is the one the prover found, not a minimal
one, and an axiom outside it may still be useful in another proof. Seeds give different proofs, hence different
cores; each distinct core of a problem becomes one training example.

### join

Inputs are collect directories, CSV files in the older shard format (which `prover_cores/shard.py` still writes), or
joined datasets; to add results later, join the earlier dataset with the new ones. `DATASET/data.npz` stores each
distinct graph of a problem once and one example per distinct core; `metadata.json` lists the inputs, provers and
seeds. The file also records which source found each core and every outcome, proved or not, which training ignores.
`--max-example-mb` of `train` compares the size of each graph's original CSV row.

### train

Writes `DATA_DIR/model/epoch-NNNN.ts` after every epoch and copies the newest to `model.ts`. There is no early
stopping; choose the epoch afterwards from the held-out metrics. `--resume --epochs N` continues to N epochs in total
and reproduces an uninterrupted run exactly; it refuses a different dataset, split, seed or network. `--evaluate 8`
scores part 8 after every epoch (`--evaluate-every`) without training on it. `--validation-run-ratio R` additionally
runs guided proof search on 1/R of those problems in the background; the larger runs do this afterwards with
`experiments/backfill_validation_runs.py` instead and pass 0.

| Network | Width / message rounds / hidden layers |
| --- | --- |
| `SmallFull` | 32 / 2 / 1 |
| `DefaultFull` | 64 / 3 / 2 |
| `LargeFull` | 128 / 4 / 3 |
| `ExtraLargeFull` | 256 / 5 / 4 |

Each size also has a `NoTerms` variant (no term and variable nodes) and a `NoComplements` variant (no edges between
complementary literals). A network is a file in `models/` defining a class of the same name with its
`default_config`; adding a file adds a `--network` choice.

### run

Prints one CSV row per problem; with `--output FILE` the rows go to the file, and repeating the command resumes it.
Without `--model` or `--sine` the search is the unguided baseline. With a checkpoint, alternative clauses are tried in
a weighted random order, each clause weighing its predicted probability (floored at `1e-6`, raised to `1 / T` with
`--temperature T`), conjecture clauses weighing 1; the seed fixes the order. `--sine` weighs the clauses that basic
SInE selects by 1 and the others by the floor, as a control without a model. A plain TorchScript scorer without
training metadata, such as a random or symbol-overlap control, also works as `--model`. `--policy` defaults to the
policy that labelled the checkpoint's training data.

## Native SATCoP

`--policy SatCoP` needs the [`axiom_selection` branch](https://github.com/mfixman/satcop/tree/axiom_selection) of
SATCoP (`external/satcop`), built with a PicoSAT that has `--trace`, which the clause-core API requires:

```bash
curl -fL https://fmv.jku.at/picosat/picosat-965.tar.gz | tar -xz -C /tmp
(cd /tmp/picosat-965 && ./configure.sh -O --trace && make -j4 libpicosat.a)
cp /tmp/picosat-965/libpicosat.a external/satcop/
(cd external/satcop && RUSTFLAGS="-L native=$PWD" cargo build --release)
```

`collect` finds the executable through `--satcop-location` (a checkout or the file itself, default `~/TP/satcop`), `run`
through `$SATCOP_LOCATION`. Each worker is pinned to one CPU, so SATCoP starts a single search thread with exactly
the given seed. `satcop --collect` prints its matrix and the matrix clauses of the PicoSAT core; the labels are over
SATCoP's own clausification, with its definitions, Skolem symbols and equality axioms. For guided search, `run` first
asks for the matrix alone (`--collect --clausify`), scores it, and passes one weight per clause with `--weights`.
Only the wall-clock timeout applies. A run that crashes leaves its log in `DATASET/native/` or a `satcop-run-*`
temporary directory.
