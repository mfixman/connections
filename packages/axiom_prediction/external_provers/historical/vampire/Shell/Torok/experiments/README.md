# Torok transfer experiments

Measured outcomes and the proposed HPC comparison are in [RESULTS.md](RESULTS.md).
The follow-up targeting standalone SInE, the fresh local Prover9 comparison,
and the late-Torok schedules are in [SINE_FALLBACK.md](SINE_FALLBACK.md).

These experiments use the existing model without retraining. `compare.py` gives
each strategy a single 60-second wall deadline including executable startup,
model loading, graph construction, inference, and proof search. A worker pins
its processes to one CPU; paused baseline processes retain memory but do not
search concurrently with their guided alternative. Proof logs, executable/model
hashes, source snapshots, manifests and per-case JSON results are retained.

The default experiment uses untyped CNF/FOF problems marked Theorem,
Unsatisfiable or ContradictoryAxioms in TPTP. MD5 family partitioning matches
Connections scheme 5. Variants stay in the same partition. This is a development
split, **not** separation from model training: the checkpoint used scheme 4,
training parts 0–7 and validation part 8. Errors and scoring failures remain
in the denominator. A successful result requires a refutation SZS status;
failure of an incomplete search is never counted as satisfiability.

## Native experiments

`--torok_file Shell/Torok/torok.ts` alone keeps the original weighted queue.
The following environment settings instead preserve Vampire's ordinary queues:

* `TOROK_SUPPORT=K` with `--sos all`: select the highest-scoring K input axiom
  clauses as additional support. Other axioms remain active/usable. Conjecture
  clauses remain support. Later preprocessing descendants of selected axioms
  inherit support membership. Stable input order breaks score ties.
* `TOROK_FILTER=K`: retain the highest-scoring K axiom clauses and all conjecture
  clauses, removing the rest. The problem is explicitly marked incomplete.
* `TOROK_SOFT=A`: multiply normal clause-selection weight by
  `1 + A * (1 - score)`, rounded upward. Retain the ordinary age/weight and
  split queues and every clause. A descendant inherits its highest-scoring
  parent's score; conjecture score is one. This bounds the penalty between
  one and `1 + A`, instead of replacing selection with random sampling.
* `TOROK_RANDOM_SUPPORT=1`: shuffle the ranking with fixed seed 1, still charging
  neural inference cost. For soft weighting, permute the scores among axioms.
  This is an ablation, not a second trained model.

With SInE enabled, scoring happens after SInE filtering and clausification.
The normal path with an empty `--torok_file` is unaffected by these variables.
Support and filtering are experimental alternatives; do not set both variables.
Soft weighting is also an alternative, not a combination with support/filtering.
`TOROK_MAX_CLAUSES=N` rejects scoring before graph construction when the
clausified input has more than N clauses. The wrapper then resumes baseline.
The runner exposes this as `--max-clauses N` (zero disables the guard).

## Comparison runner

```sh
python Shell/Torok/experiments/compare.py \
  --output results/support-development --split 100 --part 0 --workers 18 \
  --seconds 60 --modes baseline sine included resume-support1 resume-support16
python Shell/Torok/experiments/summary.py results/support-development
```

Modes `baseline`, `sine` and `included` use no model. `sos` uses Vampire's
ordinary conjecture-only support. `support1`, `support16`, `filter16`, and
`filter64` are standalone guided runs. Other integer cutoffs are accepted.
`weighted` selects the original weighted queue.
`support1-soson` uses ordinary literal selection for background axioms instead
of selecting all their literals; `noavatar-filter16` disables AVATAR in that
guided search. These are separate experimental modes, not new defaults.
`soft4` and `soft16` select bounded score penalties; `soft0` charges inference
without changing clause weights.

`resume-support1` runs baseline for `--probe` seconds (default 1), pauses it,
tries support for at most `--fraction` of the total budget (default 1/3), and
resumes the original baseline if needed. The guided allowance includes scoring.
If baseline proves the problem during the probe, guidance is skipped. An early
guided failure returns its unused time to baseline. `resume-filter16` works
similarly. Prefix `sine-` or `included-` applies that SInE setting to both stages;
prefix `random-` requests the ranking ablation.
For example, `resume-sine+support1` pauses unfiltered baseline, tries SInE and
then learned support, and resumes baseline if both fail. The two alternatives
split the guided allowance equally; every stage shares the original deadline.
Explicit stage durations are also accepted, e.g.
`resume-included:4+filter16:10` gives SInE-included four seconds and filtering
ten seconds. A prefix beginning `resume-` preserves the unfiltered baseline;
`included-resume-...` instead uses SInE-included for the baseline too.
`sinebase-resume-filter16:10` starts with SInE-included, then scores the original
unfiltered problem during its guided attempt. This can recover axioms removed
by SInE; the `included-resume-...` prefix keeps SInE enabled in both stages.
With `--probe 60 --seconds 60`, guidance runs only if the initial search exits
without a proof before the deadline. Thus `sinebase-resume-filter16:10+support16:60`
leaves a running SInE search uninterrupted, and uses only its remaining time
after failure. Compare against both `included` and
`sinebase-resume-baseline:60` to distinguish model gains from ordinary fallback.

`--algorithm otter` is the initial controlled comparison. Explicitly pass
`--algorithm lrs` to compare against Vampire's usual saturation algorithm.
All arms in one invocation use the same algorithm, seed, memory limit and
extra options. The baseline's internal Vampire timer uses the full budget;
the parent enforces the shared wall deadline across stages. Guided processes
receive their actual slice budget as their internal time limit.

Paused baseline processes use `TOROK_ACTIVE_LRS=1`. Their initial probe uses
the unchanged wall-time estimate. After SIGCONT, LRS estimates throughput and
maintenance cost from active process CPU time, excluding the pause. Remaining
time and the external deadline still use wall time. This prevents a pause from
being interpreted as slow proof search. Ordinary standalone runs are unchanged.

The first two development batches predate this pause accounting and use the full
60-second internal limit for guided slices. Their saved snapshots preserve that
protocol; do not merge them with corrected-schedule measurements silently.

Check the optional clock independently:

```sh
c++ -std=c++17 -O2 -I. Shell/Torok/experiments/check_clock.cpp \
  Shell/Torok/SearchClock.cpp -o /tmp/torok-check-clock
/tmp/torok-check-clock
/tmp/torok-check-clock enabled
```

For HPC partition jobs, pass a distinct output path for each `--part`, set
`--workers` to the CPU allocation, and use the same modes/settings across jobs.
The runner resumes completed cases only when the manifest, executable and model
match. It snapshots locally built binaries; build on the HPC before running.
The current executable needs the LibTorch libraries used at build time.

The summary reports gains and losses on paired completed cases. Its exact
discordance test treats problems independently, so correlated TPTP variants and
repeated tuning limit its interpretation. Report family-level results and new
partition validation before making broader claims.

`python Shell/Torok/experiments/attribution.py results/RUN --reference included`
reports which search stage produced each additional proof. An initial-search
gain in a resume arm occurred before model inference and must not be attributed
to the learned selector.
