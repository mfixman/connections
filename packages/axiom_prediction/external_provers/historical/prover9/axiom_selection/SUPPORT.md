# Torok-guided set of support

This is the configuration to take forward. It gives a small development-set
coverage improvement over ordinary Prover9, not a general speedup.

## Fixed rule

1. If there are at most 16 non-conjecture clauses, run ordinary Prover9.
2. Otherwise give ordinary Prover9 one second of search.
3. If it has not proved the problem, score the input once with Torok. Put the
   16 highest-scoring axioms and every conjecture clause in `formulas(assumptions)`
   (the initial set of support). Put every other axiom in `formulas(usable)`.
4. Search with ordinary Prover9 and the remaining time. If this search terminates
   without a proof, retry all clauses in the original configuration with whatever
   time remains. Scoring has a five-second cap; on scoring failure, use that same
   ordinary-Prover9 fallback.

No axiom is deleted. The selected support determines which initial clauses start
inference chains; usable clauses remain available as inference partners. Prover9
also skips its initial predicate elimination when the usable list is nonempty.
This is consequently a search-configuration change, not merely a reorder.
A finite support-restricted run can miss proofs; failure is never reported as
satisfiability. The experiment showed actual regressions as well as gains.

The numerical choices are **16 axioms** and a **one-second preliminary search**;
the five-second scoring cap is a resource guard. Ranking is deterministic, with
input order breaking equal scores. Absolute probability calibration is not used.
The same native Prover9 executable runs both arms. The old weighted queue and
its derived-clause score inheritance are not used by this configuration.

## Measurements

The main run used 180 randomly sampled known-unsatisfiable CNF families from
Torok's recorded split scheme 4, part 8. This is development data, including the
partition used to select the checkpoint, not an independent final test.
One variant per family was selected. Two common unsupported-symbol inputs were
excluded, leaving 178 problems. Every arm had a 30-second wall limit, including
parsing, per-problem graph construction, inference, and all search stages.
Model loading and three warm-up forwards occur once per worker; this measures
persistent batch inference, not cold standalone invocations.

| Configuration | Proofs / 178 | PAR-2 seconds |
|---|---:|---:|
| Ordinary Prover9 | 101 | 26.916 |
| Torok support, top 16 | 102 | 26.852 |
| Random support, scoring cost charged | 101 | 27.167 |
| Random support, no neural inference | 102 | 26.534 |

PAR-2 charges twice the limit for an unsolved problem. Torok gains **one net
proof**, approximately 1% more proofs, and improves PAR-2 by only about 0.24%.
On the 99 problems both configurations prove, summed time is **181.48 seconds
for Torok versus 148.86 for baseline**. Thus the coverage gain comes with
slower common successes; it is not a broad speed improvement.

| Problem | Baseline seconds | Torok seconds |
|---|---:|---:|
| GEO077-4 | timeout | 19.83 |
| GEO083-1 | timeout | 1.92 |
| HWV003-1 | timeout | 16.35 |
| SET025-8 | 11.65 | timeout |
| SET163-6 | 10.49 | timeout |

All five gain/loss classifications reproduced in three additional runs each.
These repetitions confirm those cases' stability; they do not enlarge the
independent problem sample or establish general superiority. A separate random-support control without neural inference also solved 102
problems: six gains and five losses versus baseline. Thus this experiment does
not establish that the learned ranking is better overall than random support.
Torok has fewer regressions and proves HWV003-1, which random support does not.
The no-inference control ran separately, so its wall-time differences are less
well controlled than the paired main comparison.

Torok reached an actual selected-support search on 46 problems in the main run,
proving eight. Other runs finished in the preliminary/default search, fell back
after scoring failure, or hit the overall deadline. Model timeouts remain part
of the measured outcome; they are not excluded as invalid inputs.

Full inputs, complete first proofs, predictions, timings, and manifests are in:

- `results/paired-support16/` — main paired run, 488.57 seconds, 17 workers.
- `results/support16-repeats/` — three repeats of each discordant problem.
- `results/support16-random-control/` — random support without model inference.

The fixed decisions and implementation hashes are recorded in
[`support_config.json`](support_config.json). The final rebuilt native Prover9
executable has exactly the same SHA-256 as the measured executable. Export parity,
CNF/equality checks, disabled-adapter proof parity, and the first-proof runner
checks all pass.

## Further development comparisons

Two subsequent comparisons reused the same 180-family manifest (178 supported
inputs), with 18 workers and 30 seconds per arm. They are repeated measurements
on the same development sample, not additional independent test data.

| Run | Baseline | Preserved top-16 | Alternative |
|---|---:|---:|---:|
| `weighted-support16` | 101 | 104 | Weighted sampling: 103 with either seed |
| `half-support-comparison` | 101 | 104 | Half-budget baseline then top-1: 104; longer scoring cap: 104 |
| `resume-support-comparison` | 101 | 103 | Half-budget baseline then top-16: 103; short top-16 then resume baseline: 103 |

Weighted sampling draws 16 distinct axioms with probability proportional to
Torok scores. It did not beat deterministic ranking. Allowing scoring up to a
quarter of the budget also did not improve coverage. The preserved strategy's
count varies across runs: LCL806-1 lies near the scoring cutoff, and NUM017-1
lies near the search deadline. Compare alternatives with the baseline and
preserved strategy measured in the same run, rather than the original 102.

The half-budget strategy pauses the initial native search before scoring and
restarts inference with top-1 support. If scoring or the guided search fails
early, it resumes the saved native process. Its PAR-2 was 26.297 seconds versus
26.314 for top-16 and 26.908 for baseline: effectively tied with top-16, with
different gains and regressions. It retains the paused prover's memory while
running scoring and guidance, so its peak memory can be higher. Each case still
has only one CPU-active search or scorer at a time.

Short top-16 followed by resumed baseline preserved all 101 baseline successes,
but only added GEO083-1 and LCL806-1. PAR-2 worsened to 27.088 seconds versus
26.895 for baseline. Preserving successes alone did not make it a faster strategy.

These results are recorded in each directory's `summary.json`; generate them
with `python axiom_selection/experiments/summary.py RESULTS_DIRECTORY`.

## Preserved command-line strategy

```sh
python axiom_selection/compare.py --strategy support16 --seconds 30 \
  --manifest axiom_selection/results/paired-support16/manifest.json \
  --workers 18 --output axiom_selection/results/support16-rerun
```

This explicitly selects the previous top-16 rule and compares it with baseline.
The previous implementation is also archived in `strategies/support16_snapshot.py`.

## Running a longer comparison

```sh
python axiom_selection/build.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python axiom_selection/experiments/paired.py \
  --all-cnf-unsat --seconds 60 --minimum 16 --probe 1 --workers 18 \
  --modes Baseline TorokSupport RandomSupport \
  --output axiom_selection/results/full-cnf-comparison
```

`--all-cnf-unsat` selects all known-unsatisfiable CNF variants in the sibling
TPTP checkout. This includes model-training families, so report that fact and
separate the recorded held-out families for generalization claims. Alternatively,
pass `--manifest PATH`, a JSON object with a `paths` array, to supply an exact
problem list. Without either option the runner samples development part 8.

**Scope:** this adapter currently accepts CNF only. FOF requires a shared,
separately checked clausification stage. Some quoted/numeric/distinct-object
encodings are rejected explicitly. Do not interpret unsupported inputs as
proof-search failures or this experiment as a result for all TPTP languages.

The model is loaded once per worker. The defaults are 60 seconds per arm and one single-core worker per CPU in the
process affinity mask. Use `--workers 18` explicitly for local experiments. Each problem/configuration is isolated with a strict wall
deadline and process-group cleanup. The manifest records the model, executable,
source hashes, paths, budgets, and choices.

## C and Python responsibilities

`native_search.c` launches Prover9, streams its proof output, enforces the search
deadline, and stops at the first complete refutation. This avoids waiting for
Prover9's automatic multiple-denial search to finish several proofs. Its JSON
records wall and child CPU time. Both arms use this same runner.

Prover9's search itself is C. Python handles the experiment, bundled parser/graph construction, and calls into the compiled TorchScript model.
Rewriting the small selection loop would save very little: its measured median
was about 0.025 milliseconds on the initial development sample. Graph construction
and model inference remain material costs on large inputs. The final model is
loaded and warmed once per worker, with one Torch thread, and per-problem scoring
is charged inside the deadline.

```sh
python axiom_selection/tests/first_proof.py
python axiom_selection/experiments/paired_summary.py \
  axiom_selection/results/paired-support16
```

Earlier filtering, soft-weight, tie-breaking, and `auto2` experiments are retained
under `experiments/`. They are not part of the fixed rule. Historical results
using the final `THEOREM PROVED` marker have a different stopping condition and
must not be pooled with these first-refutation measurements.
