# Frozen-model experiments, 26 October 2026

The best development result is 98/129 proofs against default LRS Vampire's
94/129: four additional proofs, a 4.26% relative increase (3.10 percentage
points). A late SInE-only fallback matches this count. This is a promising
schedule, but it is **not evidence of a 4.26% improvement from the model**.
The learned selector gives repeatable improvements on individual problems;
its aggregate advantage has not reached the requested 3% threshold.

## Protocol and provenance

Every arm has a 60-second end-to-end wall budget, including model loading,
graph construction and inference. Each worker uses a single pinned CPU, with
one active search at a time. Alternative arms for each problem use the same
CPU, with rotated execution order. The final runs use native seed 1, LRS,
2 GiB per-process memory, and the existing TorchScript checkpoint without
retraining. Proof success requires a refutation SZS status; errors count as
failures. Full proof logs and per-case JSON are under the listed result paths.
Elapsed times can exceed 60 seconds slightly while killing and reaping a
process; no additional search time is intentionally allocated.

Checkpoint SHA256:
`76e6274eedb3d8530886a0d4c69c2e190934890d7b8da87ae05ac0e0be2217b4`.
Starting revision: `a1332fd5bf973200ed0c21f40214767710650e7f`.
Each run snapshots its executable, checkpoint, runner, Torok sources and
tracked source diff. These experiments have uncommitted native changes.

The scheme-5 split-100 part-0 development sample contains 129 known-provable
untyped CNF/FOF problems from 91 families (70 CNF, 59 FOF). It is not held out
from training: 77 families belong to checkpoint training partitions, seven
to validation, and seven to the held-out partition under scheme 4.

## Completed development comparisons

| Experiment | Baseline | Alternative | Proofs | Relative change |
| --- | ---: | --- | ---: | ---: |
| Initial Otter, 129 problems | 87 | Original weighted queue | 35 | -59.77% |
| Initial Otter | 87 | Resume support 1 | 87 | 0.00% |
| Initial Otter | 87 | Resume support 16 | 84 | -3.45% |
| Early LRS, 129 problems | 94 | Soft penalty 4 / 16 | 85 / 84 | -9.57% / -10.64% |
| Early LRS | 94 | Resume filter 16 / 64 | 94 / 93 | 0.00% / -1.06% |
| Early LRS | 94 | Resume SInE axioms + support 1 | 96 | +2.13% |
| Corrected late LRS, 129 problems | 94 | Model filter 16 | 94 | 0.00% |
| Corrected late LRS | 94 | SInE included | 98 | +4.26% |
| Corrected late LRS | 94 | SInE included + model filter 16 | 98 | +4.26% |
| Prover9 development sample, 180 problems | 125 | Model filter 16 | 127 | +1.60% |
| Prover9 development sample | 125 | SInE included + model filter 16 | 127 | +1.60% |

Result directories respectively: `results/torok-support-pilot0`,
`results/torok-lrs-candidates0`, `results/torok-late0`, and
`results/torok-prover9-dev`. Their `summary.txt` files contain paired results.

The initial two batches predate the corrected pause accounting and guided
internal deadlines described below. They motivate follow-up experiments;
they should not be merged with corrected runs. The initial Otter batch also
did not enforce the same CPU across arms of each problem.

The corrected late schedule spends up to 45 seconds on ordinary Vampire,
pauses the still-running process, tries SInE included for four seconds and/or
learned top-16 filtering for ten seconds, then resumes the original search
if time remains. Model scoring is skipped above 256 preprocessed clauses.
Filtering retains all conjecture clauses and the selected axiom clauses;
it marks the transformation incomplete, so unsuccessful search cannot
establish satisfiability.

For the 129-problem run, combined-schedule gains are FLD060-3, SET708+4,
SET770+4 and SWB063+1, with no baseline losses in that run. All four new
proofs came from the SInE stage. SInE alone instead gains SET737+4 in place
of FLD060-3. Model-only filtering gains FLD060-3 but loses BOO007-1.
PAR2 is 35.566 seconds for baseline, 33.316 for SInE fallback and 33.344 for
the combined schedule. Four wins and no losses give exact discordance
p = 0.125, before accounting for tuning and correlated variants; the pilot
does not establish statistical significance.

The 180-problem comparison reuses the Prover9 development manifest in
`~/TP/Prover9/axiom_selection/results/resume1-comparison/manifest.json`.
It belongs to checkpoint validation part 8, not held-out part 9. Vampire
includes all 180 problems, including two omitted by Prover9's parser.
Model-only gains are NUM017-1 and SET149-6. NUM017-1 was proved by the
ordinary baseline probe, so it is timing variability rather than a learned
proof. Combined gains are LCL430-1 from SInE and SET149-6 from the model.
Neither schedule loses a baseline proof in this measurement.

## What the model demonstrably helps

Early filtering ablations use a three-second baseline probe, ten-second
guided allowance and the same scoring cap. The random control pays the
same inference cost but shuffles the ranking using one fixed seed. Native
proof-search seeds 1 and 2 are repeated; these are not independent random
subsets.

| Problem | Baseline seconds, seeds 1 / 2 | Learned filter 16 | Random filter 16 |
| --- | --- | --- | --- |
| FLD060-3 | timeout / timeout | 7.49 / 7.81 | timeout / timeout |
| SET149-6 | 58.09 / 42.86 | 4.80 / 5.40 | 58.33 / 46.98 |

Both learned results are guided proofs. SET149-6 random-control proofs
come from resumed ordinary Vampire after the filtered attempt fails.
Its development baseline timed out, whereas these repeats succeed;
the reproducible benefit is a large speed improvement, with unstable
60-second coverage. Logs are under `results/torok-fld-ablation`,
`results/torok-fld-ablation-seed2`, `results/torok-set-ablation`, and
`results/torok-set-ablation-seed2`.

## Failure analysis and implementation

Replacing Vampire's queue with weighted random selection performs badly.
Bounded neural penalties preserve its queues but still lose too many proofs.
The Prover9-style additional-support transplant alone also fails to improve
aggregate performance. Hard filtering is more useful on individual examples,
but its incomplete search must share time with a reliable fallback.

Pausing LRS exposed a separate measurement problem: its throughput estimate
counted the pause as slow search and changed pruning dramatically. An opt-in
clock now retains ordinary wall accounting before the first resume and uses
active process CPU time for search-throughput estimates after SIGCONT.
The overall deadline and remaining budget remain wall time. Guided attempts
also receive their actual slice as the native deadline, rather than 60 seconds.
KLE029+1 recovers from an erroneous resumed timeout to 14.70 seconds after
this correction (standalone baseline: 4.13 seconds).

Normal unpaused Vampire keeps its original clock and queues. Most new code is
grouped in `Shell/Torok`: `Support`, `Soft`, `SearchClock`, and this experiment
directory. Existing integration points are small. Build, clock checks,
deadline/cleanup checks and native filtering/weighting smoke checks passed.

LRS remains sensitive to timing even with a fixed seed and CPU. A subsequent
four-problem schedule-order diagnostic loses BOO007-1 despite its baseline
finishing in 57.41 seconds. Late fallbacks therefore do not guarantee zero
regressions. Reversing the order to model first has only diagnostic evidence,
not a full-sample result.

## HPC follow-up

The strongest aggregate candidate is the late fallback schedule. Keep its
SInE-only control to measure whether the learned stage adds anything:

```sh
python Shell/Torok/experiments/compare.py \
  --output results/late-part0 --algorithm lrs --workers 18 \
  --seconds 60 --probe 45 --max-clauses 256 --split 100 --part 0 \
  --modes baseline resume-included:4 resume-filter16:10 resume-included:4+filter16:10
python Shell/Torok/experiments/summary.py results/late-part0
```

Build Vampire with LibTorch on the HPC first. Use distinct output directories
for partitions, and keep the combined worker count within the allocated CPUs
(18 total on this workstation). A larger run should report paired gains and
losses, proof stage, family coverage and repeated seeds, rather than only a
single total. Training-held-out validation is recorded separately below.

## Checkpoint-held-out validation

The completed validation samples 120 CNF families uniformly from the 462
eligible known-provable families in scheme-4 part 9, with one random variant
per family and sampling seed 20261026. This partition was excluded from the
recorded checkpoint training and validation. It is not a claim that these
problems have never been examined in earlier research.

| Schedule | Proofs | Paired change from baseline | PAR2 seconds |
| --- | ---: | ---: | ---: |
| Baseline | 86/120 | — | 35.960 |
| Late filter 16 | 86/120 | 0 | 35.952 |
| Late SInE included | 86/120 | 0 | 35.920 |
| Late SInE included + filter 16 | 85/120 | -1 | 36.646 |
| Late filters 8, 16, 32, five seconds each | 80/110 | 0 on these 110 | 34.887 |

The extra three-cutoff arm was run on 110 of the 120 problems; its paired
baseline also solves 80/110. There are no new proofs from any fallback in
this validation. The combined schedule loses SWW407-1. Its baseline finishes
in 38.09 seconds, but the combined arm's baseline probe is still unresolved
at 45 seconds and neither alternative succeeds. This illustrates the timing
variability and the risk of diverting the final seconds of ordinary search.

The model-only arm attempts guidance on 34 problems; 14 exceed the 256-clause
guard and return immediately to baseline. The remaining attempts produce
no guided proofs. The resource guard limits applicability, but raising it
also risks consuming most of the short guided budget constructing graphs.

These results do not reproduce the development improvement. The +4.26%
schedule is therefore a development candidate for a larger paired HPC
evaluation, not a demonstrated general improvement. Source manifests,
complete logs and the merged summary are in `results/torok-heldout9`.
