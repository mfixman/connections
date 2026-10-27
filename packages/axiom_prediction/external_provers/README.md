# Prover9 and Vampire experiment handoff

Status: 27 October 2026. Continue the cross-prover axiom-selection research from
this Connections checkout. Prover9 ResumeSupport1 has a measured HPC coverage
improvement. Vampire has repeatable individual learned-filtering speedups, but
no demonstrated aggregate held-out advantage from Torok.

The native integrations, models, and runners remain in `~/TP/Prover9` and
`~/TP/vampire`; this directory contains documentation and compact evidence.
No new proof runs or HPC jobs were started for this handoff. The latest saved
local batches completed on 26 October. An SSH attempt on 27 October failed to
authenticate, so current remote state and older Vampire HPC outcomes are unknown.

## Reading order and evidence

* Read this file for the consolidated results and next comparisons.
* Read [SESSION_NOTES.md](SESSION_NOTES.md) for the Codex session history,
  corrections to earlier claims, and detailed exploratory screens.
* [historical/prover9/axiom_selection/](historical/prover9/axiom_selection/)
  preserves README, RESULTS, SUPPORT, RESUME, and the frozen configurations.
* [historical/vampire/Shell/Torok/](historical/vampire/Shell/Torok/) preserves
  the integration README and the experiment README, RESULTS, and SINE_FALLBACK.
* [evidence/](evidence/) preserves the latest local sample manifests, summaries,
  stage attribution, HPC paired summary, and compact audit findings.
* [sources.json](sources.json) records original paths, byte counts and SHA-256
  hashes for copied files, raw-result pointers, and relevant session rollouts.

Historical files are byte-for-byte snapshots. Commands and result links inside
them refer to their original repositories; relative raw-result links may not
resolve inside this documentation copy. They also retain superseded claims and
old output-format descriptions. Use the consolidated account and session
corrections here when interpreting them. Current code and a run's frozen
manifest take precedence over historical reproduction instructions.

The 451 MB Prover9 HPC archive, model files, binaries, and per-problem logs stay
in their original locations. Full session transcripts are not copied: they
include unrelated conversation and authentication material.

## Model and evaluation protocol

Torok is the frozen ExtraLargeFull checkpoint at epoch 45: width 256, five
message-passing rounds, four hidden layers, ReLU. It was trained using BCE on
observed SATResetCoP SAT-core membership. A negative label means absent from
that observed core, not useless for every proof. No transfer experiment below
retrained the model. Exported model SHA-256:

```text
76e6274eedb3d8530886a0d4c69c2e190934890d7b8da87ae05ac0e0be2217b4
```

Checkpoint splitting used scheme 4 with ten parts: training 0–7, checkpoint
validation 8, held-out 9. Family grouping keeps related filename variants
together. Scheme-5 partitions used to distribute jobs do not demonstrate
separation from training. Changing the split count under scheme 4 changes
membership; a tenth of old part 9 must be selected within that existing part.

The main local development sample has 180 CNF families from checkpoint part 8;
Prover9 initially excluded two unsupported-symbol inputs, leaving 178.
Vampire's two successive 120-family held-out samples are different:

* `torok-heldout9`: sampling seed 20261026, drawn from 462 eligible families.
* `torok-sine-failure-validation`: seed 20261028, drawn from 337 remaining
  eligible families after excluding preceding experiment families. This is the
  sample used for the final local cross-prover comparison.

Both are excluded from checkpoint training, but repeated examination and
strategy evaluation mean neither is now an untouched final test set.
The mixed 129-problem development sample uses scheme-5 split-100 part-0 and
contains 70 CNF and 59 FOF problems from 91 families; it overlaps training.
It differs from the October 21 part-1 comparison subset of 129 problems.

All percentages below are relative changes in proof counts. PAR2 charges twice
the deadline to an unsuccessful run; lower is better. Keep errors in reported
denominators and also report common unsupported inputs separately. Unproved
searches are not satisfiability certificates. Count family coverage alongside
variant coverage. Unions of separate full-budget runs are not fixed-budget
portfolio measurements.

## What the integrations change

| Intervention | Prover9 | Vampire |
| --- | --- | --- |
| Weighted given-clause queue | Four weighted picks per native pick; geometric parent-score inheritance | Replaces ordinary clause selection with score-weighted sampling; geometric parent-score inheritance |
| Support selection | Selected axioms and conjectures in support, remaining axioms usable; a nonempty usable list disables initial predicate elimination | Selected axioms added to conjecture support, with remaining axioms as background; literal-selection settings also matter |
| Filtering | Removes unselected axioms for that attempt | Keeps top-k axioms plus conjecture clauses; incomplete transformation |
| Soft weighting | Modifies ordinary clause weights or breaks ties | Preserves ordinary queues and applies bounded score-dependent penalties |

These are integration heuristics, not rules learned by the checkpoint.
AVATAR is Vampire's SAT-based splitting mechanism. Disabling it is an
independent search change, so an ordinary AVATAR-off control is essential.
The Vampire comparisons use a single LRS or Otter strategy, not a CASC portfolio.

## Prover9 local development

| Test | Baseline | Alternative | Interpretation |
| --- | ---: | ---: | --- |
| Original weighted queue, 122 problems, 15s | 61 | Torok seeds 1/2: 38/44; uniform: 40/43 | No Torok gain over baseline; replacing the native selection mixture itself hurts |
| Filtering, 178 supported problems, 20s | 99 | Probe + top-64/score-threshold filter: 97; direct filter: 96; random: 95 | Filtering did not improve coverage |
| Initial support-16, 178 problems, 30s | 101 | Learned 102; random with inference cost 101; random without inference 102 | Small support benefit does not isolate learned ranking |
| Weighted support-16 comparison, same development manifest, 30s | 101 | Deterministic 104; weighted seeds 1/2 both 103 | Weighted subset sampling did not beat deterministic ranking |
| Half-budget support-1 / longer scoring cap, 30s | 101 | Half-budget top-1 104; top-16 104; longer scoring cap 104 | Different proof sets, same coverage |
| Half-budget or resumed support-16, 30s | 101 | All guided alternatives 103 | Resumed support-16 retains baseline successes but worsens PAR2 |
| Final ResumeSupport1, 30s | 101 | Preserved top-16 103; ResumeSupport1 106 | Five net gains, no baseline losses in this run |

Counts of preserved top-16 vary from 102 to 104 across batches because scoring
and search cutoffs affect borderline cases. Compare contemporaneous paired arms.

Support-16 gives ordinary search one second, then places the top 16 axioms and
conjectures in support if there are more than 16 axioms. It allows five seconds
for scoring, and can retry ordinary search if guidance terminates early.

ResumeSupport1 gives ordinary search one second, pauses it, scores with a cap
of one-quarter of the total budget, runs top-1 support for at most one-third of
the total budget, and resumes the saved process if necessary. Thus at 30s the
scoring cap is 7.5s and guided search cap 10s; at 60s they are 15s and 20s.
Parsing, scoring and all search stages share one wall deadline. One warmed,
single-thread model per worker is loaded outside individual problem timing.
Only one search or scorer is CPU-active per case, but a paused process retains
memory. All other axioms remain usable.

The final 30s comparison has PAR2 26.840s baseline, 26.598s support-16 and
26.351s ResumeSupport1. Shared successes take 157s versus 341s in total for
baseline and ResumeSupport1. The improvement is coverage rather than a general
speedup. Four gains reproduced as timeout distinctions; GRP235-1 became a
roughly 28s baseline success versus 6s guided success in repeats. There is no
matching random-support control for the final schedule.

## Vampire local development

On October 21, the part-1 split-100 comparison attempted 192 inputs, excluded
63 explicitly unsupported inputs, and retained runtime failures and three
parser errors in the 129-input comparison. At 60s per problem, ordinary Vampire
proved 86, SInE-included 70, SInE-axioms 56, Torok weighted 34,
included-plus-Torok 37 and axioms-plus-Torok 32. Ordinary Vampire proved every
Torok success. A separate 20-problem training-partition sample gave ordinary
15, SInE 13, Torok 4 and combined 5.

October 26 development used a different mixed 129-input sample:

| Test | Baseline | Alternatives |
| --- | ---: | --- |
| Initial Otter | 87 | Weighted 35; resumed support-1 87; support-16 84 |
| Early LRS | 94 | Soft penalties 4/16: 85/84; resumed filter-16/64: 94/93; SInE-axioms + support-1: 96 |
| Corrected late LRS | 94 | Model filter-16 94; late SInE-included 98; late SInE + model 98 |

The corrected late policy gives ordinary search up to 45s, tries SInE for 4s
and/or filtering for 10s, then resumes if time remains within 60s. Model scoring
is skipped above 256 preprocessed clauses. On development 180, model-only and
combined alternatives each proved 127 versus baseline 125. On the first
held-out sample of 120, baseline, model-only and SInE-only each proved 86;
combined proved 85. No fallback produced a new proof on that held-out sample.

### SInE failure recovery

Let SInE run until it finishes or exhausts the shared minute. Only after an
early unsuccessful exit, retry the original unfiltered input. The filter/support
variant allows 10s filtering then support with remaining time; the
filter/ordinary variant allows 6s filtering then ordinary search.

| Configuration | Development /180 | Second held-out /120 |
| --- | ---: | ---: |
| Standalone SInE-included | 91 | 64 |
| SInE then ordinary Vampire | 110 | 77 |
| SInE then learned filtering/support | 109 | 75 |
| Same filtering/support with shuffled ranking | Not measured | 76 |
| SInE then learned filtering/ordinary fallback | 111 | 78 |

Every fallback arm retained the standalone SInE proofs in these measurements.
The one-proof lead of filter/ordinary over ordinary fallback was not caused by
a learned-stage proof in either sample. This supports fallback after SInE's
early failures, but not superiority of learned ranking over the controls.

### Final baseline-first AVATAR comparison

Run ordinary LRS Vampire for up to 45s; pause it; try SInE-included for up to
4s; try top-16 filtering with AVATAR disabled for up to 10s including scoring;
resume ordinary search if time remains within 60s. Keep the 256-clause cap.

| Fallback after the ordinary probe | Development /180 | Mixed CNF/FOF /129 |
| --- | ---: | ---: |
| SInE for 14s | 126 | 98 |
| SInE 4s then ordinary AVATAR-off 10s | 127 | 99 |
| SInE 4s then learned filter-16 AVATAR-off 10s | 129 | 101 |
| Same learned schedule with shuffled ranking | 129 | Not measured |

Learned and shuffled development arms have identical proof sets. None of the
three gains over the mixed-sample SInE control came from the learned stage.
LCL247-3 and LCL319-3 have only nine and ten axioms, so top-16 retains everything.
Ordinary AVATAR-off proves both. Treat this as an experimental schedule, not a
demonstrated learned-ranking improvement.

Individual ranking ablations are more positive: learned filter-16 proves
FLD060-3 in 7.49/7.81s while baseline and shuffled filters time out under two
native search seeds. SET149-6 takes 4.80/5.40s with learned filtering versus
ordinary proofs around 43–58s; shuffled filtering fails and relies on resumed
ordinary search. These are development examples, not aggregate validation.

## Latest local cross-prover comparison

All arms use the same pinned CPU per problem and a separate 60s wall deadline.
The 180 development and second 120 held-out manifests are unchanged. Prover9
warms its model per worker and uses 512 MB native memory; Vampire loads its
model inside the attempt and uses 2,048 MB. This preserves the tested
configurations rather than equalizing memory and startup policy.

| Configuration | Development /180 | Held-out /120 | Total /300 | PAR2 s |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Prover9 | 106 | 68 | 174 | 52.686 |
| Prover9 ResumeSupport1 | 107 | 70 | 177 | 52.488 |
| Prover9 baseline 50s then Torok 10s | 106 | 70 | 176 | 52.109 |
| Prover9 baseline 40s then Torok 20s | 107 | 70 | 177 | 51.752 |
| Vampire final baseline-first schedule | 129 | 83 | 212 | 38.075 |

The late Prover9 allowance includes scoring; these schedules do not resume
baseline after guidance. If baseline exits early, guidance can start early but
still gets at most its stated allowance. The late schedules ran in a separate
follow-up batch. Both lose GRP186-2 and SWV330-1, which baseline proves at about
58s. The 10s allowance misses NUM017-1; both late schedules miss SCT040-1, whose
scoring and guided search need about 25s. They do not improve coverage beyond
ResumeSupport1. Four unsupported-symbol inputs remain in the full denominator;
excluding them gives Vampire 208 versus ResumeSupport1 177.

Of Vampire's 212 proofs, 208 are ordinary-stage, one SInE-stage, and three
filtered AVATAR-off-stage. Only SET149-6 involves nontrivial top-16 selection
in those three. All 83 held-out proofs came from ordinary Vampire. Standalone
ordinary Vampire was not run on that same held-out sample, so the final
schedule's 83 does not establish a benefit over ordinary search. All 1,500
outcomes across the two local batches agree with retained proof markers;
source and executable hash checks found no changes during the runs.

## Prover9 HPC comparison

The saved run measures 6,269 distinct CNF problem filenames once under each
strategy, with 60s per arm. Ten job partitions use scheme 5. All 18,807 CSV
proof flags agree with the original archived JSON in the saved audit; no
missing or duplicate problem/strategy pairs were found. This verifies recorded
outcomes, not independent certification of every proof.

| Strategy | Proofs /6269 | Gains | Losses | Net vs baseline | PAR2 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3825 | — | — | — | 48.803 |
| ResumeSupport1 | 3921 | 142 | 46 | +96 (+2.51%) | 47.639 |
| WeightedSupport16 | 3812 | 97 | 110 | -13 (-0.34%) | 49.053 |

WeightedSupport16 samples a support subset of 16 distinct axioms proportional
to scores. It is neither deterministic top-16 filtering nor the original
weighted given-clause queue. All 142 ResumeSupport1 gains have successful
guided-stage proofs in the archive. Shared successes take about 49.8% more
total time than baseline. There is no matching random-support control.

| Actual checkpoint partition under scheme 4 | Inputs | Baseline | ResumeSupport1 |
| --- | ---: | ---: | ---: |
| Training parts 0–7 | 5049 | 3068 | 3147 |
| Validation part 8 | 623 | 385 | 387 |
| Held-out part 9 | 597 | 372 | 387 |

Held-out part 9 has 17 gains and two losses: +15 (+4.03%). It represents 462
families; family coverage rises from 297 to 304. The full denominator includes
56 common input errors, six in held-out part 9. On the exact local development
180 filenames, the HPC baseline/ResumeSupport1 counts are 113/112; on the
second held-out 120 they are 70/72. Do not substitute those for the latest local
106/107 and 68/70: hardware, timing and measured outcomes differ.

Original saved results:

```text
~/TP/Prover9/axiom_selection/results/hpc-three-way-60s.csv
~/TP/Prover9/axiom_selection/results/hpc-three-way-60s.tar.gz
~/TP/vampire/results/prover9-crosscheck/
```

Older Vampire sessions discuss SInE/Torok HPC runs across ten parts at 60s per
problem, but no completed local HPC outcome table was found. An October 20
build job ID is not evidence that a proof experiment completed. Reconcile
remote scheduler records and logs after authentication before reporting results.

## Reproduction and next comparisons

Read each external repository's current AGENTS.md before editing it. Preserve
the saved results and frozen manifests; use new output paths for changed
policies. Root [AGENTS.md](../../../AGENTS.md) and [plan.txt](../plan.txt)
govern new work. This handoff is not authorization to start experiments,
submit jobs, resume monitoring or pull on the HPC.

The missing comparisons are:

1. Prover9 baseline versus ResumeSupport1 versus the identical random-top-1
   support schedule, charging the same scoring cost. Include late-20 if useful.
2. Vampire ordinary baseline versus final schedule versus matching model-free
   and shuffled controls on the same fresh training-excluded families.
3. Stage attribution, paired gains/losses, family counts, scoring cost and
   input failures for each run. Repeat native seeds without treating repeats
   or variants as independent new families.

Keep 18 single-core local proof workers and total local experiment usage within
19 cores. Retain one end-to-end wall deadline per strategy; do not grant each
stage a fresh full minute. Pin alternatives to the same CPU and rotate order.
Preserve warm/cold model policy in the manifest or explicitly change it for all
arms. Keep the original and corrected timer/first-proof protocols separate.

These commands read existing results only, from this Connections directory:

```sh
python ../Prover9/axiom_selection/experiments/summary.py \
  ../Prover9/axiom_selection/results/resume1-comparison
python ../vampire/Shell/Torok/experiments/local_prover9_summary.py \
  ../vampire/results/torok-local-prover9-final \
  --late ../vampire/results/torok-local-prover9-late
```

For future authorized proof runs, use the native runners in their own checkout
with explicit absolute dataset/output paths, or a subshell changing into that
checkout. There is no unified cross-prover launcher in Connections yet.
