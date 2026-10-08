# Command-line reference

```bash
python axiom_predictor.py COMMAND --help
```

## Commands

| Command | Purpose |
| --- | --- |
| `collect [PROBLEM ...]` | Collect resumable SAT-core labels in --dataset PATH or DATA_DIR/dataset. |
| `train [PROBLEM ...]` | Train on saved labels; explicit problem inputs collect fresh labels first. |
| `evaluate [CHECKPOINT] [PROBLEM ...]` | Evaluate saved labels, or collect fresh labels for explicit problems. With --model or --data-dir, every positional input is a problem; otherwise the first is the checkpoint. |
| `predict [CHECKPOINT] [PROBLEM ...]` | Print axiom rankings and probabilities as JSON lines. |
| `run [PROBLEM ...]` | Run SATCoP/SATResetCoP and print per-problem JSON results. A selected checkpoint enables weighted guidance; otherwise runs the baseline. |

PROBLEM accepts .p files, flat directories, TPTP filenames, or category codes.
Without problem inputs, collect/run/predict select all supported FOF/CNF
candidates under the TPTP root. Training without inputs uses the saved dataset.
CHECKPOINT is a model.pt file or its directory. For fresh evaluation,
put the explicit checkpoint before the problem paths, or select the checkpoint
with --model or --data-dir and supply only problem paths.

## Arguments

| Argument | Commands | Default and meaning |
| --- | --- | --- |
| `-h`, `--help` | All | Show help. Root and training help list the available network classes. |
| `--data-dir PATH` | All | Required for train; required for collect only without --dataset. Holds model/ or models/NAME/ and the default dataset/. For evaluate it selects the checkpoint and default dataset; for predict/run it selects a checkpoint. |
| `--dataset PATH` | collect, train, evaluate | Override DATA_DIR/dataset. Collect writes resumable JSONL shards directly here; train/evaluate read the shared dataset without copying or modifying it. Cannot accompany PROBLEM inputs in train/evaluate. |
| `--network NAME` | train | **DefaultFull**. Select a Python file/class from models/. The optional .py suffix is accepted. |
| `--model PATH` | run, evaluate | Explicit trained checkpoint or directory. Otherwise use --data-dir; without either, run the unguided baseline. |
| `--model-name NAME` | train, evaluate, predict, run | Optional saved-run name: use DATA_DIR/models/NAME rather than DATA_DIR/model. Separate from the network implementation. Existing checkpoints require --resume. |
| `--resume` | train | Continue the selected checkpoint, restoring optimizer and RNG state. Requires the same data and training configuration; --epochs is the total target, not additional epochs. Older checkpoints without training state cannot resume. |
| `--tptp PATH` | All | Optional explicit corpus root. Otherwise discover it as described below. |
| `--split N` | All | **1**. Partition by a stable hash of the problem prefix, such as `ABC123`; variants stay together. |
| `--parts P ...`, `--part P ...` | All | All parts. Select zero-based parts. Keep training/validation/test parts disjoint with the same N. |
| `--evaluate PART ...` | train | Score these held-out parts during training, e.g. `--split 10 --parts 0 1 2 3 4 5 6 7 --evaluate 8`. Overlapping or empty evaluation selections fail. |
| `--evaluate-every N` | train | **1**: evaluate every epoch. Positive N evaluates every N epochs after the first evaluation. Set 0 to measure evaluation cost and space evaluations to target 10% extra training time. First/final epochs always evaluate. |
| `--seed N` | train, run | **0**. Training initialization/shuffle seed, or proof-search seed. Does not change the split or fresh-label collection's internal search seed. |
| `--epochs N` | train | **200** passes through the training examples. Each epoch is retained as `epoch-0001.pt`, `epoch-0002.pt`, etc. in the model directory; `model.pt` is an identical copy of the latest checkpoint. All include optimizer/RNG state. |
| `--batch-size N` | train, evaluate | **43 problems**, not clauses. Optimizer-update batch size in training; prediction batch size in evaluation. Memory depends on graph sizes. |
| `--log-every N` | train | **10**. Report training-set prediction metrics every N epochs plus first/final epochs. Zero disables intermediate metric evaluations. Loss and checkpoint saving happen every epoch, independently of this option. |
| `--device DEVICE` | train, evaluate, predict, run | **cuda**, failing when CUDA is unavailable. auto selects CUDA if available and CPU otherwise; cpu forces CPU; cuda:N selects a visible device. Baseline search does not use a neural device. |
| `--num-workers N` | collect, train, evaluate, run | Maximum available CPUs, bounded by SLURM allocation and affinity, except guided CUDA search defaults to one worker. Processes for collection/search, CPU threads for training/evaluation. Processes are also capped by problem count. Explicit N overrides automatic selection. |
| `--policy SatCoP\|SatResetCoP` | collect, train, evaluate, run | Inherit the checkpoint policy for run/evaluate, or dataset policy for training; otherwise **SatResetCoP**. Explicit conflicts with recorded checkpoint/dataset policies fail. |
| `--step-limit N` | collect, train, evaluate, run | **1,000,000,000 prover steps per problem for collect; 1,000,000 prover steps per problem for other commands**. Only affects fresh search, not optimization or saved-label evaluation. |
| `--timeout-seconds S` | collect, train, evaluate, run | **900 seconds (15 minutes) per problem for collect; 120 seconds for other commands** for fresh search. Collection and run supervise workers, including startup, parsing and any worker-side checkpoint loading/prediction. Overdue workers are terminated, including with one worker. Not the total job duration. |
| `--wandb`, `--no-wandb` | train, evaluate, run | Tracking **on** by default. Missing package/credentials or startup failure produces a warning and execution continues. --no-wandb disables it quietly. |

Use --model-name for multiple experiments sharing one dataset. An explicit
CHECKPOINT or run --model cannot be combined with --model-name. Network
architecture, inputs, and activation are defined in the selected Python
class. Adam defaults to learning rate 0.001 and weight decay 0; these are not
CLI options.

With --dataset, collect writes CATEGORY.jsonl for a single category/directory,
or problems.jsonl for explicit files, multiple inputs, or the default corpus.
Resumable intermediate records live in PATH/.cache. Do not run simultaneous
collectors writing the same shard; finish collection before training readers.
Existing dataset directories and individual JSONL files remain readable.
Do not mix legacy metadata.json/examples/ records and public JSONL shards in
one directory: the loader rejects this instead of silently ignoring records.
Shard compatibility ignores the machine-specific TPTP installation path;
policy and collection budgets must still match. Use the same corpus version
on each machine: ignoring its path does not verify its contents.

Resuming on another allocation does not require the same number of visible
GPUs. Only the training device's CUDA RNG is saved; CPU training saves none.
Changing CPU/GPU or hardware is supported but need not be numerically identical.

Fixed choices use TitleCase StrEnum members and values internally. Policy
arguments also accept the older lowercase spellings (satcop/satresetcop).
Checkpoints and dataset metadata retain their existing lowercase strings;
loading them converts those values to enums where used by configuration.

## Four HPC experiments

Set these paths on the HPC and use the policy that collected the dataset:

```bash
export EXP_DATASET=/path/to/shared/dataset
export EXP_TPTP=/path/to/TPTP
export EXP_POLICY=SatResetCoP
```

Run each training command as its own GPU job. Defaults are seed 0, 200 epochs,
batch size 43, CUDA required, and W&B enabled with graceful startup fallback.
The commands read the same dataset and write separate run directories:

```bash
python axiom_predictor.py train --dataset "$EXP_DATASET" \
  --data-dir runs/DefaultFull --network DefaultFull --policy "$EXP_POLICY" \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py train --dataset "$EXP_DATASET" \
  --data-dir runs/SmallFull --network SmallFull --policy "$EXP_POLICY" \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py train --dataset "$EXP_DATASET" \
  --data-dir runs/DefaultNoTerms --network DefaultNoTerms --policy "$EXP_POLICY" \
  --split 10 --parts 0 1 2 3 4 5 6 7

python axiom_predictor.py train --dataset "$EXP_DATASET" \
  --data-dir runs/DefaultNoComplements --network DefaultNoComplements \
  --policy "$EXP_POLICY" --split 10 --parts 0 1 2 3 4 5 6 7
```

Optional fifth: repeat with LargeFull for both --network and the run directory.
To resume an interrupted run, repeat its exact command with --resume.

Evaluate held-out SAT-core labels on validation part 8:

```bash
for network in DefaultFull SmallFull DefaultNoTerms DefaultNoComplements; do
  python axiom_predictor.py evaluate --dataset "$EXP_DATASET" \
    --data-dir "runs/$network" --split 10 --parts 8
done
```

After freezing the experiment choices, compare proving on test part 9.
These commands cover the entire eligible test partition of the specified
TPTP corpus, including problems without collected labels. Use the same
corpus version and any category restriction for every run. One worker avoids
giving the baseline more CPU parallelism than the guided runs.

```bash
mkdir -p results
python axiom_predictor.py run --tptp "$EXP_TPTP" --policy "$EXP_POLICY" \
  --split 10 --parts 9 --seed 0 --num-workers 1 \
  --timeout-seconds 120 --step-limit 1000000 > results/Baseline.jsonl

for network in DefaultFull SmallFull DefaultNoTerms DefaultNoComplements; do
  python axiom_predictor.py run --tptp "$EXP_TPTP" --policy "$EXP_POLICY" \
    --data-dir "runs/$network" --split 10 --parts 9 --seed 0 --num-workers 1 \
    --timeout-seconds 120 --step-limit 1000000 > "results/$network.jsonl"
done
```

Output redirection replaces an existing results file; choose a new filename
when preserving an earlier run. Compare proved counts first, then timings;
validation prediction metrics alone do not establish improved proving.

## Network names

- SmallFull, SmallNoComplements, SmallNoTerms
- DefaultFull, DefaultNoComplements, DefaultNoTerms
- LargeFull, LargeNoComplements, LargeNoTerms

See [models and extension example](README.md#models) for their sizes, input
definitions, and how to add a network class.

## TPTP discovery

1. Explicit --tptp (an invalid path fails).
2. The TPTP environment variable (an invalid path fails).
3. TPTP/ in the working directory.
4. TPTP/ in the repository root.
5. TPTP/ beside the repository.
6. corpora/TPTP* under the repository and then the working directory,
   with higher numeric version names considered first.
7. ~/TPTP.

Automatically discovered roots must contain both Problems/ and Axioms/.
Standalone files remain usable without a corpus. If no root can be found,
benchmark-name resolution reports an error asking for a TPTP root.
The resolved root on this checkout is the sibling ../TPTP.

## Structured output

All commands write machine-readable records to stdout and progress logs to
stderr. Output defaults to JSON Lines. Add `--csv` before or after the command
to use CSV with one fixed header for that command, including on empty output.
CSV streams records as they become available. Strings are ordinary CSV cells;
numbers and booleans use JSON syntax, and lists/dictionaries are JSON-encoded
inside cells. An absent field is empty; an explicit null is `null`.

Train and evaluate emit multiple event types. Run emits only per-problem
results, without an `event` column. Final aggregate metrics are stored in
`PATH.resume/completion.json` when using `--output`, or printed to stderr
otherwise; they are not rows in CSV or JSONL output.
Collect emits its collection summary. Predict emits one row per axiom.
Filter by `event` when processing streams with different record types.
The `problem` field (and prediction output’s `problem_path`) contains the
filename including its extension, without its directory (e.g. `SYN001-1.p`).
Full paths are retained in resume checkpoints.
Run's per-problem results include `part`, calculated using the active `--split`.
Run, prediction, evaluation problem rows, and training's `event=model` include
`network_size`: the number of feedforward scorer parameters (weights and biases),
excluding the GNN encoder. NoModel reports 0.
Evaluate also emits `event=problem` rows containing `problem`, `part`, and
per-problem metrics, including rows for skipped problems. The `outcome`
column is replaced by `tptp_status`, copied from the problem file's `% Status :`
declaration (e.g. `Theorem` or `Satisfiable`). It is blank when the declaration
is missing or the file is unavailable. This is the declared status, including
when proof search times out; `proved` still records proof-search success for run. W&B per-problem tables include `part` too. Without an
explicit split, `part` is 0. Aggregate summaries still combine all selected parts.
The same records and fields are emitted with and without `--csv`; CSV headers
include the union of possible fields, with irrelevant cells left empty.
Saved checkpoints, dataset shards, and metrics files retain their existing formats.

```bash
python axiom_predictor.py predict --csv --device cpu model.pt SYN088-1.010.p > predictions.csv
```

## Fixed behavior

Guided search uses weighted ordering, temperature 1, and every axiom. There
are no --mode, --temperature, or --top-k options. Both the root help and
train help enumerate the network implementations from their Python files.

W&B uses team mfixman-phd-team and project axiom-prediction. Training runs use
the network name (for example, DefaultNoTerms); resumes add the total invocation
count: DefaultNoTerms-2, DefaultNoTerms-3, and so on. The count is stored in
training_runs.json in the model directory, including invocations with tracking
disabled. Existing checkpoints without a count are treated as having run once.
Standalone evaluation runs use the checkpoint's network name followed by
`-evaluate` (for example, `DefaultNoTerms-evaluate`).
Use `--run-name NAME` on train, evaluate, or run to set an exact W&B name,
overriding automatic prefixes and suffixes.
Grouping uses the saved-run name. Credentials come from WANDB_API_KEY
or secrets/wandb_key relative to the working directory. Install optional
tracking with python -m pip install wandb. Only the on/off CLI switches remain.

`run/ratio_problems_proved` records proved problems divided by processed problems,
including unsuccessful attempts in the denominator. It updates after each result
and in the final summary, including saved results when resuming. With no processed
problems, the final ratio is null.

No automatic multi-GPU training occurs. Without --multiprocess, each guided-search
worker loads its own model. Guided CUDA search then defaults to one worker;
larger explicit worker counts share the selected GPU and may exhaust its memory.
With --multiprocess, workers use one shared model as described below.

Problems without conjecture clauses use an empty conjecture set, pooled to a
zero vector, during training, evaluation, and guided search. An empty axiom set
produces no predictions and contributes no training loss or ranking metrics.
If an entire training batch has no axioms, no optimizer step is taken. Metrics
without any labels are null.
Parsing failures print the problem and diagnostic to stdout and set error to
true. Use --output to keep saved CSV/JSONL records separate from these messages.
Unexpected search errors produce a nonzero command exit status.

Evaluate reports SAT-core label prediction, not improved theorem proving.
Only successfully labelled problems contribute to these metrics. Use run on
the full held-out problem list to measure proof success, including failures
from collection. Training reports training-set metrics; `--evaluate` adds held-out
SAT-core label metrics under `evaluation/*` to the same W&B epoch upload.
Evaluation uses cached graphs and labels (fresh PROBLEM inputs collect labels once).
With `--evaluate-every 0`, the automatic interval measures inference plus metric computation against training
time; first/final evaluations and initial label collection can exceed the 10% target.
Batch-progress uploads do not trigger evaluation. Evaluation also works with
`--no-wandb`, writing `evaluation_metrics.json` beside the checkpoint. Final
per-problem and ranked-axiom tables use `evaluation_results/*` in W&B.
`train/macro_average_precision` and `evaluation/macro_average_precision` average
per-problem AP with equal weight, excluding problems with no positive labels
(undefined AP). If all problems have undefined AP, the metric is omitted from W&B.
The run summary records `evaluation/best_macro_average_precision` and
`evaluation/best_epoch`, retaining the earliest epoch on ties. These summarize
evaluations within the current W&B run; a resumed training invocation starts a
new run. Every epoch's checkpoint is retained locally; `model.pt` selects the
latest epoch for prediction and resuming. W&B uploads the final model.
These are prediction metrics, not guided proof-search success rates.

Splits always hash the problem prefix: `ABC123+1.p` and `ABC123-2.p`
belong to the same part. Non-TPTP filenames use their name without the
directory or extension. Evaluation warns about checkpoints using older
split schemes, whose part assignments may differ.

## Parallel run and evaluate on one GPU

Add `--multiprocess` to `run` or `evaluate`. CPU worker counts follow the
allocated CPUs (or `--num-workers`). Guided runs keep one shared model on the
selected GPU; CPU workers parse and search concurrently, submitting graphs to
that model. Requests arriving together are combined into prediction batches.
Evaluation collects proofs and builds graphs in CPU processes, then scores
those graphs in adaptive GPU batches. Training and prediction commands do not
accept this flag.

Batch sizes start at 1, double after success, and bisect the last successful
and failed sizes after CUDA OOM (for example 8, 16 OOM, 12). Failed batches are
retried without dropping results. Every batch is checked because graph sizes
vary. A run problem that cannot fit alone reports an error; evaluation stops
with the OOM. This tunes problem count, so it cannot guarantee a fixed GPU
memory footprint or eliminate per-problem timeouts. Run timeouts include
parsing, waiting for shared inference, and proof search.

```bash
python axiom_predictor.py run --model runs/DefaultFull/model/best-epoch-119.pt --multiprocess --splits 10 --parts 8 9
python axiom_predictor.py evaluate --model runs/DefaultFull/model/best-epoch-119.pt --multiprocess --splits 10 --parts 8 9
```

Each command uses one GPU. No additional Python dependencies are required.

## Output files and automatic resume

`run` and `evaluate` accept `--output PATH`. Results go to that file instead
of stdout, as JSON Lines by default or CSV with `--csv` (the extension does
not select the format). Progress and errors still go to stderr.

```bash
python axiom_predictor.py run --model runs/DefaultFull/model/best-epoch-119.pt --multiprocess --splits 10 --parts 8 9 --output results/run.jsonl
python axiom_predictor.py evaluate --model runs/DefaultFull/model/best-epoch-119.pt --multiprocess --splits 10 --parts 8 9 --csv --output results/evaluate.csv
```

Repeat the same command after interruption to resume. Completed run outcomes,
including timeouts and unreadable problems, are retained and skipped; final
aggregate metrics include both previous and new results. An atomic
`PATH.resume/completion.json` marker makes a subsequent invocation a no-op,
preserving a run's failure exit status. The marker records the output size,
exit status, and aggregate metrics. Older CSVs with an `outcome` column (or `run` CSVs with an `event` column)
have an incompatible header; use a new output path for those runs.
An incomplete final record is removed
before appending, and simultaneous writers to the same file are rejected.

Keep the companion `PATH.resume/` directory with the output. It stores command
settings and, for evaluation, collected examples/skips and prediction scores.
Evaluation resumes collection and scoring from those checkpoints and computes
aggregate metrics across the entire selection. Work interrupted before a
checkpoint is committed is repeated. Settings must match the previous command;
worker count, device, multiprocessing, tracking name and tracking on/off may change. To rerun
completed failures or change the experiment, choose a new output path.

Without `--output`, the existing stdout behavior is unchanged. Redirected old
stdout files may contain an HPC `nvidia-smi` preamble; they are not clean
result files and must not be used directly as resume files.

### SInE comparison

Use `python axiom_predictor.py run --sine PROBLEM --multiprocess` for CPU
SInE guidance, or `python axiom_predictor.py evaluate --sine PROBLEM` to compare
its selections against freshly collected SAT-core labels. Saved labels can be
scored with `evaluate --sine --dataset PATH` or `evaluate --sine --data-dir DIR`
(which reads `DIR/dataset`). All positional evaluation arguments are problem
inputs when `--sine` is set. No checkpoint or GPU is needed. `--sine` conflicts
with `--model` and `--model-name`; `run --sine` also rejects `--data-dir`.

This is basic **clause-level** SInE with tolerance 1, unlimited expansion depth,
and generality threshold 0. Predicate, function and constant symbols count once
per axiom clause; variables do not count. Conjecture symbols activate clauses
through their least frequent symbols, repeatedly to a fixed point. Clauses with
no symbols have no triggers; a conjecture with no symbols selects nothing.
Saved graph evaluation uses the same algorithm without reopening problem files.

Search retains every clause. Selected axioms and conjecture clauses receive
weight 1, while unselected axioms retain the existing `1e-6` floor in weighted
guidance at temperature 1. Output rows use `mode=sine` and network size 0.
Evaluation uses exact binary scores with the existing metrics: BCE treats these
decisions as probabilities (using the metric's numerical clipping), and tied
top-k scores follow existing clause-order tie breaking. Saved dataset label
policy is inherited; fresh labels use `--policy` or the usual default.

Splits, `--output` resumption, CSV and W&B reporting work as usual. Keep SInE,
model and unguided results in separate output files; their resume identities
are incompatible. SInE summaries record its fixed settings and have no model
configuration or training split.
