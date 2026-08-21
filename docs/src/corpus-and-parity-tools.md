# Corpus And Parity Tools

`connections/tools/` contains developer commands for validating and measuring
the prover. These tools are not part of the public `connections` API, and they
are not imported by the library package.

## Corpus Runner

Download benchmark corpora with:

```bash
connections-download-benchmarks --list
connections-download-benchmarks tptp-v6.4.0 iltp qmltp --root benchmarks
```

Use `--all` to download every known corpus and `--force` to replace existing
targets. Archives are cached under `ROOT/.downloads` by default.

Run pycop over a corpus slice with:

```bash
pycop PATH --out artifacts/corpus/runs.jsonl --overwrite
```

It selects problem files from one or more paths, runs pycop over each problem,
and writes one JSONL row per problem. It also writes a summary sidecar by
default: `artifacts/corpus/runs.summary.json`.

Useful options:

```bash
pycop Problems/SYN \
  --out artifacts/corpus/syn.jsonl \
  --pattern "*.p" \
  --limit 25 \
  --settings cut \
  --settings "comp(7)" \
  --steps 1000 \
  --timeout 10 \
  --continue-on-error \
  --overwrite
```

Use `--summary-out PATH` to choose the summary path, or `--no-summary` when
only JSONL rows should be written.

The output row schema is intentionally small and stable:

- `problem`
- `path`
- `status`
- `outcome`
- `szs_status`
- `inference_actions`
- `elapsed_seconds`
- `strategy_count`
- `winning_strategy_index`
- `error_type`
- `error_message`

The runner delegates to `connections.runs.run_corpus`, which accepts a
schedule, source directories, and problem paths. The `pycop` CLI supplies
pycop strategies and output-file handling. Downstream experiment code should
import `connections.runs`, not developer tool modules.

## Policy Experiment Entry Points

`run-pycop` selects one policy explicitly while retaining convenient TPTP
codename, raw-problem, directory, SLURM-slice, and parallel-corpus handling:

```bash
run-pycop SYN001+1.p \
  --tptp "$TPTP" \
  --strategy FirstActionIDPolicy \
  --max-steps 1000 \
  --timeout 10
```

`--strategy` is a compatibility alias for `--policy`.  A policy can be a
built-in alias or an import path such as
`my_package.policies:LearnedPolicy`. Result rows are JSON objects containing
both total policy calls and inference-action counts.

Use `compare-strategies` for a problem-by-policy experiment matrix:

```bash
compare-strategies Problems/SYN \
  --strategies FirstActionIDPolicy learned=my_package.policies:LearnedPolicy \
  --settings cut \
  --name syn-policies \
  --num-workers 8
```

The command writes a human progress report, a final CSV, and a JSONL partial
file.  The partial file starts with a manifest covering the code revision,
source and policy implementation hashes, problem hashes, settings, budgets, and
slice. A run resumes only when that manifest matches, preventing results from
an older prover revision or configuration from being silently reused. Use a
new `--name` or pass `--overwrite` to start a fresh run.

The bundled Connections policies can be compared directly:

```bash
compare-strategies SYN001+1.p \
  --strategies LeanCoPCon LeanCoPCon_2 SATCoPCon SATResetCoP \
  --name connections-policies
```

- `LeanCoPCon` is first-action iterative deepening with leanCoP cut and
  `comp(7)`.
- `LeanCoPCon_2` uses the same policy class across the bundled 30-phase
  leanCoP 2.1 schedule.
- `SATCoPCon` accumulates ground tableau clauses in an incremental CaDiCaL
  shadow and uses its model for literal selection and model lemmata.
- `SATResetCoP` uses the same SAT control and resets the tableau at dead ends
  instead of locally backtracking.

The SAT policies accept a result from their SAT shadow only when the grounded
clause set is UNSAT. An ordinary tableau closure is restarted until the shadow
validates it.

`--split` and `--part` select modulo slices. If neither is supplied, Slurm
array values are read from `SLURM_ARRAY_TASK_COUNT` and
`SLURM_ARRAY_TASK_ID`. Worker count similarly defaults from
`SLURM_CPUS_PER_TASK` when present.

The summary schema is:

- `schema`
- `output`
- `summary_output`
- `problems`
- `theorem`
- `unsatisfiable`
- `counter_satisfiable`
- `satisfiable`
- `timeout`
- `gave_up`
- `error`
- `inference_actions`

## Boundary

Reusable corpus selection, run rows, summaries, and generic prover execution
belong in `connections.runs`. `pycop` owns pycop-specific CLI argument
parsing, progress printing, and output-file plumbing.

## Profiling

Profiling is available through the `pycop` CLI and uses `connections.runs`
profile helpers internally:

```bash
pycop PATH --profile artifacts/profile --overwrite
```

It selects problem files, runs pycop under `cProfile`, and writes:

- `profile.pstats`
- `runs.jsonl`
- `profile_functions.jsonl`
- `profile_callers.jsonl`
- `profile_overview.json`
- `summary.json`

When no `--settings` tokens are supplied, the profiler uses the built-in pycop
schedule for the selected logic. When settings are supplied, it profiles that
single pycop strategy:

```bash
pycop Problems/SYN \
  --profile artifacts/profile/syn-cut-comp7 \
  --settings cut \
  --settings "comp(7)" \
  --steps 1000 \
  --timeout 10 \
  --overwrite
```

The profiling logic is reusable package code because learning experiments may
profile their own prover runs with the same run-row artifacts.

## Parity

Reference-prover and parity tools live under `tools/parity`. They are
developer diagnostics, not package APIs:

```text
tools/
  parity/
    reference_provers/
    prefix_oracle.py
    run_all.py
    run_prefix_parity.py
    run_prefix_equation_parity.py
    run_free_variable_admissibility_parity.py
    run_matrix_parity.py
    run_trace_parity.py
    run_status_check.py
```

They use bundled leanCoP, ileanCoP, and mleanCoP code as correctness oracles.
These tools may require SWI-Prolog and corpora, so they should remain
explicit developer commands rather than normal pytest tests.

The three corpus-level checks are:

- **status**: run native `pycop` and compare its SZS status with the benchmark
  ground-truth label when one is available.
- **matrix**: compare native parsing plus matrix construction with the
  leanCoP-family translator output.
- **trace**: compare native search events with the bundled reference prover
  search events.

The lower-level prefix, prefix-equation, and free-variable admissibility checks
are oracle diagnostics for the constraint machinery used by matrix construction
and proof search.

Reference Prolog assets are copied under:

```text
tools/parity/reference_provers/prolog/
```

Do not edit the copied reference-prover files. SWI-Prolog adapters and other
local parity helpers belong under `tools/parity/prolog/`.

The umbrella validation command runs the curated prefix, matrix, status, and
trace diagnostics:

```bash
python tools/parity/run_all.py --json
```

It runs the direct prefix, prefix-equation, free-variable admissibility,
source-to-matrix parity, benchmark-status checking, and trace parity commands.
Use `--only` or `--skip` for focused runs:

```bash
python tools/parity/run_all.py --only matrix --json
python tools/parity/run_all.py --skip trace --json
```

Manifest sweeps run larger configured corpus slices. The default manifest is
`tools/parity/manifests/full-0.1.json`:

```bash
python tools/parity/run_manifest.py --json
```

The source-to-matrix manifest is separate:

```bash
python tools/parity/run_manifest.py \
  --manifest tools/parity/manifests/matrix-0.1.json \
  --json
```

Manifest runs can also write durable artifacts:

```bash
python tools/parity/run_manifest.py \
  --out artifacts/parity/status/full-0.1.jsonl \
  --summary-out artifacts/parity/status/full-0.1.summary.json \
  --overwrite
```

The direct prefix-unification oracle is:

```bash
python tools/parity/prefix_oracle.py
```

Native-vs-reference prefix diagnostics are:

```bash
python tools/parity/run_prefix_parity.py
python tools/parity/run_prefix_equation_parity.py
python tools/parity/run_free_variable_admissibility_parity.py
```

Curated status and trace checks are:

```bash
python tools/parity/run_status_check.py
python tools/parity/run_trace_parity.py
```

Both can run a small file or corpus slice:

```bash
python tools/parity/run_status_check.py \
  --path Problems/SYN \
  --source-dir path/to/TPTP-root \
  --logic classical \
  --reference leancop21 \
  --settings cut \
  --settings "comp(7)" \
  --limit 25 \
  --json
```

Without `--omit-traces`, each JSON row includes both full event arrays. That is
useful for a single problem or a small slice, but corpus-scale sweeps should
write compact rows with only statuses, trace lengths, timings, match flags, and
the first difference.

The source-to-matrix command is:

```bash
python tools/parity/run_matrix_parity.py --json
```

It normalizes native matrices and reference translator output up to variable
renaming and compares both order-sensitive and multiset views of clauses.
This is the parser plus matrix-construction parity loop for the supported
classical FOF, classical CNF, intuitionistic FOF, and modal QMF slices.

Triage guidance and classification conventions are documented in
`tools/README.md` and `CONTRIBUTING.md`.
