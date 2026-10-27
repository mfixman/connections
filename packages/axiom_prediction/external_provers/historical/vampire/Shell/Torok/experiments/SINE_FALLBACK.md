# SInE failure recovery with the frozen model

SInE followed by learned filtering and ordinary Vampire gives **111/180**
development proofs versus **91/180** for standalone SInE (+21.98%). On 120 new
checkpoint-held-out families it gives **78/120** versus **64/120** (+21.88%).
Every standalone SInE proof is retained in both measurements.

Ordinary Vampire fallback alone gives 110 and 77. The extra proof in each
model-assisted comparison is not produced by learned filtering. These results
establish recovery of early SInE failures, not superiority of learned ranking
over the stronger model-free fallback or the earlier baseline-first schedules.
The learned filtering/support variant without ordinary fallback gives 109 and
75; its shuffled-ranking control gives 76 on the held-out sample.

## Policy

Run ordinary LRS Vampire with `--sine_selection included` and its full
60-second deadline. If it exits without a proof before the deadline, try the
model on the original, unfiltered problem:

1. Keep the highest-scoring 16 axiom clauses plus the conjecture, for at most
   ten seconds including scoring.
2. If that fails, try learned support of size 16 with the remaining time.
   All other axioms remain usable in this stage.

The complete sequence shares one 60-second wall deadline. A running SInE
search is never interrupted. Guidance starts only after it has already exited,
so these runs do not depend on the experimental pause-clock correction.
Scoring is skipped above 256 preprocessed clauses. Failure of a filtered
search is not treated as a satisfiability result.

This addresses a concrete failure mode: SInE sometimes removes axioms needed
for a proof and quickly returns `GaveUp`. The model then selects from the
original axioms, including those SInE removed. Retrying with ordinary Vampire
also addresses that failure mode, and is an essential control.

The second model-assisted variant tries filtering for at most six seconds,
then ordinary Vampire on the original problem with the remaining time.

## Completed development results

| Strategy | Proofs / 180 | Change from SInE | PAR2 seconds |
| --- | ---: | ---: | ---: |
| Standalone SInE | 91 | — | 60.867 |
| SInE, then ordinary Vampire | 110 | +20.88% | 49.088 |
| SInE, then learned filtering and support | 109 | +19.78% | 49.578 |
| SInE, then learned filtering and ordinary Vampire | 111 | +21.98% | 48.281 |

All arms preserve the 91 standalone SInE proofs in this measurement. The
learned filtering/support arm adds 18 proofs from 18 different families,
all in guided stages. Its sole loss against ordinary fallback is SYN260-1,
whose input exceeds the scoring cap. Some small inputs have at most 16 axioms;
for those, filtering retains all axioms and the gain does not depend on ranking.

The apparent one-proof advantage of the filtering/ordinary-Vampire variant
over ordinary fallback is **not a model gain**: SYN560-1 is proved by its
initial SInE search in 20.55 seconds, before inference is invoked. The other
arms time out on that same initial search. LRS timing variability remains
material even with fixed seeds and matched CPUs.

There are substantial model-assisted speedups on individual common proofs:

| Problem | SInE then ordinary Vampire | SInE then learned filtering |
| --- | ---: | ---: |
| SET149-6 | 53.11 s | 2.75 s |
| GEO083-1 | 19.24 s | 3.04 s |

The earlier SET149-6 ranking ablation in [RESULTS.md](RESULTS.md) also supports
a real ranking effect: learned filtering succeeds quickly under two native
search seeds, while the shuffled filter fails and relies on ordinary fallback.

Logs, snapshots and paired summaries: `results/torok-sine-failure2`.
`attribution.txt` identifies the stage producing every gain.

For context, the earlier comparison on these same 180 problems obtained 125
proofs with ordinary Vampire and 127 with baseline-first model-assisted
schedules. Those were separate runs, but the gap makes clear that preserving
SInE first is not the strongest overall strategy on this sample. This follow-up
isolates the narrower question of improving standalone SInE.

## Other completed follow-ups

On the original 129-problem development sample, putting learned filtering
before SInE gives 98 proofs. Both the four-second and fourteen-second SInE-only
fallbacks also give 98; standalone SInE gives 81. Changing the order therefore
does not establish an improvement over the stronger schedules.
Logs: `results/torok-model-first0`.

A deliberately difficult 24-problem screen allows a larger 1,024-clause
scoring cap and a full minute per standalone method. SInE solves three,
filter-16 two, filter-64 one, support-1 one, support-16 two, and SInE followed
by support-1 two. Larger inputs also cause expensive scoring and memory-limit
failures. The full comparison therefore retains the 256-clause cap.
Logs: `results/torok-hard-screen2`.

A second screen on the same 24 difficult problems retained the 256-clause cap
and varied native search settings. SInE plus filtering solved one; filtering
with AVATAR disabled solved three; soft penalty one solved two; support one
with ordinary background literal selection solved two; and support sixteen
with that literal selection solved two. These are selected diagnostic cases,
not an independent validation sample. Logs: `results/torok-hard-screen3`.

GRP186-2 was proved in 21.16 seconds by `support1-soson`, selecting one of
18 axiom clauses. This remains a single-run lead: the queued controlled
repeats were cancelled when the user requested that the current batch be the
last experiment. Do not present it as a confirmed model-specific gain.

COL046-1 initially looked promising in the same screen. Follow-up runs showed
that top-16 filtering keeps all three axioms, and ordinary Vampire solves the
problem in 2.67 seconds under both tested seeds. Disabling AVATAR without the
model solves it in about two seconds. Although learned support succeeds where
plain and shuffled support fail, it is much slower than ordinary Vampire.
This is not an improvement over the baseline. Logs:
`results/torok-col046-check-seed1`, `results/torok-col046-check-seed2`,
`results/torok-col046-filter-seed1`, `results/torok-col046-filter-seed2`.

A six-problem stage-order diagnostic gave six proofs each for ordinary
fallback, learned filtering first, and learned support first; shuffled support
first gave four. This sample was selected after observing previous outcomes,
and timing variation also changed which stage proved SET135-6. It does not
replace the frozen validation below. Logs: `results/torok-support-first-check`.

## What the configurations do

All configurations use Vampire; Prover9 supplied the motivating experiment and
one development manifest. The model has not been retrained.

| Configuration | Intervention |
| --- | --- |
| `baseline` | Ordinary single-strategy Vampire, without SInE or model guidance. These comparisons explicitly choose LRS. |
| `included` | SInE formula selection before normal proof search, using `--sine_selection included`. |
| `filter16` | Keep the learned top 16 axiom clauses and all conjecture clauses, then use ordinary search queues. Discarded axioms make failure inconclusive. |
| `support1` / `support16` | Add the learned top 1 or 16 axiom clauses to initial support alongside the conjecture. Keep other axioms as usable background. |
| `support1-soson` | The same support selection, with ordinary background literal selection (`--sos on`) instead of selecting all background literals (`--sos all`). |
| `weighted` | The original Torok probability-weighted clause queue, replacing ordinary clause selection. |
| `soft4` / `soft16` | Keep ordinary queues and penalize clause weights according to model scores. |
| `random-...` | Shuffle the axiom ranking while still paying inference cost; tests whether learned choices matter. |
| `noavatar-...` | Disable SAT-based AVATAR splitting for that attempt. |
| `resume-...` | Run ordinary Vampire first, pause if needed, try alternatives, and resume the original process if time remains. |
| `sinebase-resume-...` | Start with SInE, then run alternatives on the original unfiltered input. With a 60-second probe, fallback occurs only after SInE exits early. |

The most useful pilot result is fallback recovery of early SInE failures.
For a model-specific investigation, SET149-6 provides stronger evidence than
the aggregate counts: its ranking ablation and repeated speedups support a
real learned-selection effect. Original weighted sampling and soft penalties
did not produce competitive overall results.

## Validation protocol

The development set is the existing 180-family Prover9 sample from checkpoint
validation partition 8. It is not training-held-out evidence. Before finishing
that comparison, a new validation sample was frozen: 120 CNF families from
checkpoint-held-out partition 9, excluding families in the preceding Vampire
development and validation manifests. One variant per family is sampled with
seed 20261028 from 337 eligible remaining families.

The validation includes the four strategies above and the identical learned
filtering/support schedule with shuffled rankings. The shuffle uses one fixed
seed, still pays inference cost, and changes both filtering and support
selection. It is a ranking ablation, not another trained model.

Every arm uses native search seed 1, a 60-second end-to-end deadline, and a
2 GiB per-process memory setting. Modes for each problem share a pinned CPU;
execution order rotates between problems. At most 18 single-core workers run
across concurrent batches. The frozen model and native executable are unchanged
from [RESULTS.md](RESULTS.md); this follow-up changes the Python orchestration.

### Completed validation

| Strategy | Proofs / 120 | Change from SInE | PAR2 seconds |
| --- | ---: | ---: | ---: |
| Standalone SInE | 64 | — | 57.681 |
| SInE, then ordinary Vampire | 77 | +20.31% | 46.254 |
| SInE, then learned filtering and support | 75 | +17.19% | 48.262 |
| Same filtering/support schedule, shuffled ranking | 76 | +18.75% | 47.739 |
| SInE, then learned filtering and ordinary Vampire | 78 | +21.88% | 45.720 |

Every fallback arm retains all 64 standalone SInE proofs in this run. The
learned filtering/support arm adds 11 guided proofs from 11 families, but
the shuffled control adds 12. Ordinary fallback recovers two additional
large inputs, SYN122-1 and SYN294-1, that exceed the model's clause cap.
These results validate the fallback improvement over standalone SInE; they
do not validate superiority of learned ranking over shuffled ranking.

SET135-6 accounts for the differences between shuffled and learned support,
and between the model/ordinary-fallback variant and ordinary fallback. SInE
exits after about 47–48 seconds. The learned filter spends ten seconds without
a proof, leaving too little for support. The shuffled filter fails in about
1.3 seconds, leaving time for its support proof. In the six-second learned
variant, the proof comes from the subsequent ordinary Vampire process, not
the model. Its different native remaining-time budget can change LRS pruning;
the one-proof advantage is not evidence of a learned proof over that control.

Validation results, all proof logs and attribution are recorded in
`results/torok-sine-failure-validation`. The sample and policy were fixed
before this validation was run. Reordering stages in response to SET135-6
is a later diagnostic experiment, not part of this frozen validation.

## Final AVATAR comparison

The final batch starts with ordinary Vampire for up to 45 seconds. The control
then tries SInE for 14 seconds. Alternatives instead try SInE for four seconds,
then an AVATAR-off search for ten seconds, either on all axioms, the learned
top 16, or a shuffled top 16. An unsuccessful attempt can return unused time
to the paused baseline. All stages share the original 60-second deadline.

On the complete 180-problem development sample:

| Fallback after the baseline probe | Proofs / 180 | PAR2 seconds |
| --- | ---: | ---: |
| SInE, 14 seconds | 126 | 38.190 |
| SInE, 4 seconds; ordinary AVATAR-off, 10 seconds | 127 | 38.024 |
| SInE, 4 seconds; learned filter-16 AVATAR-off, 10 seconds | 129 | 37.235 |
| SInE, 4 seconds; shuffled filter-16 AVATAR-off, 10 seconds | 129 | 37.245 |

The learned and shuffled arms have identical proof sets: four gains and one
loss versus the 14-second SInE control, for a net +2.38%. HWV043-2 and LCL766-1
are proved by the ordinary baseline process, not by learned search. LCL247-3
and LCL319-3 are proved by the AVATAR-off stage, but filtering retains all nine
and ten axiom clauses respectively; the model-free AVATAR-off arm proves both.
COL046-1 is lost in every alternative, while the control proves it during its
initial baseline probe. This reinforces the observed timing sensitivity.
The paired exact discordance p-value is 0.375, before accounting for tuning.
There is no demonstrated learned-ranking advantage in this final comparison.

Logs: `results/torok-noavatar-development`, including `summary.txt` and
`attribution.txt`. These are development results, not a new held-out validation.

The final mixed CNF/FOF development batch also completed all 129 problems:

| Fallback after the baseline probe | Proofs / 129 | PAR2 seconds |
| --- | ---: | ---: |
| SInE, 14 seconds | 98 | 33.480 |
| SInE, 4 seconds; ordinary AVATAR-off, 10 seconds | 99 | 32.897 |
| SInE, 4 seconds; learned filter-16 AVATAR-off, 10 seconds | 101 | 31.533 |

The learned arm has three gains and no losses versus the SInE fallback control
(+3.06%; paired exact discordance p = 0.25). None is a learned-stage proof:
SCT071-1 and SWV563-1.004 are proved by the ordinary baseline process, and
SEU243+2 by the four-second SInE attempt. The model-free AVATAR-off arm also
proves SEU243+2. Thus the numerical threshold is reached on this development
sample, but there is no evidence that learned selection caused the gain.
Logs: `results/torok-noavatar-mixed0`, with paired summary and attribution.

Both final batches finished successfully, and their worker processes exited.
The two queued GRP186-2 repeats were cancelled before they started. No further
experiments are running or queued from this session.

## Cross-check against the saved Prover9 HPC run

The follow-up audit reads
`~/TP/Prover9/axiom_selection/results/hpc-three-way-60s.csv` and its companion
`.tar.gz` archive. It starts no new proof searches. There are 6,269 distinct
problem filenames, each measured once under all three strategies. All ten
partition manifests specify 60 seconds, the same model hash as Vampire, and
the same Prover9 executable hash. All 18,807 CSV proof flags agree with the
archived per-run JSON; there are no missing or duplicate problem/strategy rows.
This checks the recorded outcomes, not independent proof certification.

| Prover9 configuration | Proofs / 6,269 | Gains vs baseline | Losses | Net change | PAR2 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,825 | — | — | — | 48.803 |
| ResumeSupport1 | 3,921 | 142 | 46 | +96 (+2.51%) | 47.639 |
| WeightedSupport16 | 3,812 | 97 | 110 | -13 (-0.34%) | 49.053 |

All 142 ResumeSupport1 gains have a successful `guided_search` and an
unsuccessful `baseline_search` in the archived JSON. This is much stronger
stage attribution than the final Vampire counts: the additional proofs really
come from the support-restricted search. There is still no matching random
top-1 control for this HPC schedule, so it does not isolate learned ranking
from the effect of changing support. Common proofs take about 49.8% more total
time under ResumeSupport1 (15,817 s versus 10,562 s on the paired common set).
The improvement is primarily coverage, rather than speed on common proofs.

Repartitioning filenames by the checkpoint's actual scheme 4, rather than the
scheme-5 HPC job partitions, gives:

| Checkpoint partition | Problems | Baseline | ResumeSupport1 | Net relative change |
| --- | ---: | ---: | ---: | ---: |
| Training parts 0–7 | 5,049 | 3,068 | 3,147 | +2.58% |
| Checkpoint validation part 8 | 623 | 385 | 387 | +0.52% |
| Held-out part 9 | 597 | 372 | 387 | +4.03% |

The held-out improvement is 17 gains and two losses across 462 represented
families. Counting families with at least one proved variant instead gives
297 versus 304; variants must not be treated as independent families. The
56 common input errors in the full run, including six in held-out part 9,
remain in these denominators. Restricting to supported inputs does not change
the proof counts or net improvements.

Matching the exact Vampire sample filenames gives:

| Configuration | Development 180 | New held-out 120 | CNF portion of mixed sample: 70 |
| --- | ---: | ---: | ---: |
| Prover9 baseline | 113 | 70 | 46 |
| Prover9 ResumeSupport1 | 112 | 72 | 49 |
| Prover9 WeightedSupport16 | 110 | 69 | 47 |
| Vampire standalone SInE | 91 | 64 | — |
| Vampire SInE → learned filter → ordinary fallback | 111 | 78 | — |
| Vampire final baseline-first AVATAR-off filter schedule | 129 | not tested | 57 |

The two Prover9 gains on the new held-out 120 are MGT060-1 and SCT040-1, with
no losses. On development 180 it has no gains and loses LAT200-1. Thus the older
30-second local result of 106 versus 101 does not persist as an advantage on
that same sample under the 60-second HPC measurement. Budgets and hardware
differ, so this does not identify a single cause of that change.

The adapters also behave differently: Prover9 ResumeSupport1 pauses after one
second, allows up to 15 seconds for scoring and 20 seconds for top-1 support
search, then resumes baseline within the shared minute. All other axioms remain
usable. WeightedSupport16 samples 16 distinct support axioms proportionally to
model scores; it is neither deterministic top-16 filtering nor the original
weighted given-clause queue. Vampire's selected schedules use different stage
budgets, and filtering actually removes unselected axioms.

The cross-prover table is descriptive: Prover9 ran on the HPC with model loading
and warm-up outside per-case timing, whereas Vampire ran locally with loading
inside its deadline. Input filenames are matched; the run manifests do not
establish byte-for-byte dataset identity across machines. The Prover9 adapter
only covers CNF, so the other 59 FOF problems in the mixed sample are absent.

On held-out 120, Prover9 ResumeSupport1 proves nine problems missed by the
78-proof Vampire schedule, while Vampire proves 15 missed by Prover9. Their
union is 87, but that combines two separate 60-second runs and is not a measured
60-second portfolio. On development 180, the final 129-proof Vampire schedule
and Prover9 ResumeSupport1 have a union of 141, with the same budget caveat.

Individual cross-checks: Prover9 fails SET149-6 under all three HPC arms, while
Vampire learned filtering proved it in 2.75 seconds. Prover9 ResumeSupport1
proves FLD060-3 in 4.86 seconds while its baseline fails. For GEO083-1, Prover9
improves from 19.49 to 1.13 seconds. These timings are within-prover comparisons;
do not interpret differences between machines as controlled speed ratios.

Machine-readable audit and paired comparisons are saved in
`results/prover9-crosscheck/archive-audit.json` and
`results/prover9-crosscheck/summary.json`.

## Fresh local cross-prover comparison

At the user's request, `local_prover9.py` reruns the final Vampire schedule,
ordinary Prover9, and Prover9 ResumeSupport1 on this computer. It uses the
existing 180-problem development and 120-problem checkpoint-held-out manifests,
without changing either sample. All three arms of a problem run on the same
pinned CPU, in a deterministic shuffled order. Eighteen workers share CPUs
0–17; each arm has its own 60-second wall deadline. Native proof logs and
per-stage outcomes are retained under `results/torok-local-prover9-final`.

This compares the existing configurations: Prover9 warms a single-thread model
once per worker and charges per-problem parsing/scoring/search; Vampire loads
its model inside the guided attempt. Native memory settings remain Prover9
512 MB and Vampire 2,048 MB. It is not an equal-memory or equal-startup-overhead
microbenchmark. Common unsupported-symbol errors remain in the full denominator,
and a supported-input comparison is reported separately.

The user also requested ordinary Prover9 first, followed by a short Torok
attempt. `prover9_late.py` adds 50+10 and 40+20 second schedules. The second
stage uses learned top-1 support with all other axioms usable; its allowance
includes scoring. There is no resumed baseline stage. If ordinary search exits
early without a proof, the Torok stage can start early but still receives at
most its stated allowance. These two settings were fixed before the follow-up
started. They run in a separate batch, after the first batch releases its lock,
and reuse the same CPU assignment for each problem. Results are under
`results/torok-local-prover9-late`.

```sh
python Shell/Torok/experiments/local_prover9_summary.py \
  results/torok-local-prover9-final --save
python Shell/Torok/experiments/local_prover9_summary.py \
  results/torok-local-prover9-final --late results/torok-local-prover9-late --save
```

The combined summary includes only problem groups with all five arms complete.
The two samples are reused from preceding experiments; checkpoint-held-out does
not mean untouched by all earlier evaluation or strategy development.

The first local batch completed all 300 paired groups (900 runs):

| Configuration | Development / 180 | Held-out / 120 | Total / 300 | Overall PAR2 seconds |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Prover9 | 106 | 68 | 174 | 52.686 |
| Prover9 ResumeSupport1 | 107 | 70 | 177 | 52.488 |
| Vampire final baseline-first schedule | 129 | 83 | 212 | 38.075 |

Against ResumeSupport1, Vampire has 31 gains and nine losses on development
(net +22, +20.56%), and 19 gains and six losses on held-out problems
(net +13, +18.57%). The pooled net is +35 (+19.77%). Four problems are rejected
by Prover9's adapter for unsupported symbols: SYO630-1 and SYO676-1 in
development; SYO673-1 and SYO683-1 in held-out. Excluding them gives Vampire
127 versus 107 on 178 development problems, and 81 versus 70 on 118 held-out
problems. The lead therefore persists on supported inputs.

Of Vampire's 212 proofs, 208 come from its ordinary baseline process, one from
SInE, and three from the Torok-filtered AVATAR-off attempt. LCL247-3 and LCL319-3
keep all nine and ten axioms, respectively. SET149-6 is the nontrivial learned
selection proof: retain 16 of 112 axiom clauses, remove 96, and prove the problem
in 47.90 seconds total after the initial baseline attempt. These counts support
the whole schedule's local advantage, not attribution of that advantage to
Torok. See `results/torok-local-prover9-final/attribution.json`.

Prover9 ResumeSupport1 adds five guided proofs over its ordinary baseline:
GEO083-1, LCL806-1, NUM017-1, MGT060-1 and SCT040-1. Two baseline proofs are lost,
giving the pooled net of three. SCT040-1 spends 7.86 seconds scoring and 16.80
seconds in guided search; this illustrates why a late stage whose entire
allowance is ten or twenty seconds can miss a proof obtained by ResumeSupport1.

The late follow-up also completed all 300 paired groups (600 additional runs):

| Prover9 schedule | Development / 180 | Held-out / 120 | Total / 300 | Overall PAR2 seconds |
| --- | ---: | ---: | ---: | ---: |
| 50 seconds baseline, then up to 10 seconds Torok | 106 | 70 | 176 | 52.109 |
| 40 seconds baseline, then up to 20 seconds Torok | 107 | 70 | 177 | 51.752 |

The 40+20 schedule ties ResumeSupport1's total coverage; 50+10 is one proof
behind. Neither closes the gap to Vampire's 212 proofs. The observed PAR2 is
slightly lower for the late schedules, but they ran in a separate follow-up
batch, and a baseline-stage timing change contributes to that comparison.

Against ordinary Prover9, 50+10 gains GEO083-1, LCL806-1 and MGT060-1 through
guided search, plus SWW431-1 through its initial ordinary search. It loses
GRP186-2 and SWV330-1, for a net gain of two. The 40+20 schedule also gains
NUM017-1 through guided search, for a net gain of three. Ordinary Prover9
proves the two lost cases at 58.02 and 58.39 seconds, beyond both late schedules'
baseline cutoffs. NUM017-1 needs 12.40 seconds of guided search plus scoring:
it fits the twenty-second allowance but not the ten-second allowance.

Compared with ResumeSupport1, both late schedules lose SCT040-1 and gain
SWW431-1; 50+10 additionally loses NUM017-1. SWW431-1 is proved before Torok
runs, in 31.51 seconds under 50+10 and 22.80 seconds under 40+20. It is therefore
not evidence of a learned-selection gain. The later schedule is a reasonable
coverage/speed trade-off to test further, but this run does not establish a
coverage improvement over ResumeSupport1.

Of the 50+10 schedule's 176 proofs, 173 are baseline-stage and three guided;
of the 40+20 schedule's 177 proofs, 172 are baseline-stage and five guided.
The latter includes GEO077-4, which ordinary Prover9 also proves. All 1,500
recorded outcomes across both local batches agree with the complete-proof
markers in the retained logs, and source/executable hash checks found no
changes during the runs. Both driver processes exited; no searches remain
running or queued. Audit details are in each run's `audit.json`.

Reproduce the local batches from the Vampire repository root:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
taskset -c 0-17 python3 Shell/Torok/experiments/local_prover9.py \
  --workers 18 --output results/torok-local-prover9-final
taskset -c 0-17 python3 Shell/Torok/experiments/prover9_late.py \
  --workers 18 --previous results/torok-local-prover9-final \
  --output results/torok-local-prover9-late
```

## Reproduce on the HPC

Build the current Vampire with LibTorch on the HPC, then run separate output
directories for each partition. Include the ordinary-fallback control:

```sh
python Shell/Torok/experiments/compare.py \
  --output results/sine-recovery-part0 --algorithm lrs --workers 18 \
  --seconds 60 --probe 60 --max-clauses 256 --split 100 --part 0 \
  --modes included sinebase-resume-baseline:60 \
  sinebase-resume-filter16:10+support16:60 \
  sinebase-resume-filter16:6+baseline:60
python Shell/Torok/experiments/summary.py results/sine-recovery-part0
python Shell/Torok/experiments/attribution.py results/sine-recovery-part0
```

To compare the final baseline-first schedules, including model-free and shuffled
controls, use a separate output directory:

```sh
python Shell/Torok/experiments/compare.py \
  --output results/avatar-recovery-part0 --algorithm lrs --workers 18 \
  --seconds 60 --probe 45 --max-clauses 256 --split 100 --part 0 \
  --modes baseline included resume-included:14 \
  resume-included:4+noavatar:10 \
  resume-included:4+noavatar-filter16:10 \
  resume-included:4+noavatar-random-filter16:10
python Shell/Torok/experiments/summary.py results/avatar-recovery-part0
python Shell/Torok/experiments/attribution.py results/avatar-recovery-part0 \
  --reference resume-included:14
```

These comparisons use single-strategy LRS Vampire, not a CASC portfolio.
Positive pilot counts should not be presented as a general improvement over
default Vampire or over the best model-free schedule tested earlier.
