# Codex session history and interpretation

Recorded on 27 October 2026. Session IDs and source paths are indexed in
[sources.json](sources.json). These are research notes, not permission to start
or resume experiments. Authentication material and unrelated conversation are
excluded. Original repository AGENTS.md files are preserved as
`historical/*/AGENTS.reference.md`; those are historical references, not
instructions that override this repository's current AGENTS.md.

## Vampire integration and first comparisons, 19–22 October

Session: `01a154d4-53bf-7a03-b0ed-ac1688756b78`.

The user selected Torok as the canonical ExtraLargeFull epoch-45 checkpoint.
The requested initial integration kept every clause and used probability-weighted
given-clause selection, matching the Connections weighted-search idea. It did
not replace SInE with a learned premise filter. A tensor-only TorchScript export
enabled LibTorch inference; native graph/score parity was checked on shared CNF.
Cross-prover FOF clausification parity was not asserted.

The session set up ten family-hash input lists for HPC and commands for
SInE/Torok jobs with 60s per problem and 7h59m allocation limits. Build job
38882056 was mentioned, but it is not a proof-run completion record. The user
later said HPC jobs were taking a while and requested local testing. Completed
HPC proof counts for Vampire were not recovered in the documentation audit.

The first local split-10 run was cancelled in favour of the smaller split-100
part-1 sample. Later comparison outputs are under:

```text
~/TP/vampire/results/local-part1-split100-20261021-142003/
~/TP/vampire/results/old-part1-small-20261021-164605/
```

Important corrections made within the session:

* The initial claim that Torok rejected 77 inputs as unsupported was wrong.
  Only 63 were explicitly unsupported (34 higher-order, 29 typed first-order).
  Six allocation failures and eight undiagnosed exceptions remain failures.
  Thus the proper comparison subset is 129 of 192 attempted inputs, not 112.
* SInE `axioms` permits filtering all eligible axioms; `included` restricts
  selection to included material. The latter scored 70 versus 56 proofs, still
  below ordinary Vampire's 86 on the shared subset.
* The weighted Torok integration altered internal saturation search, including
  an untrained parent-score inheritance rule. Its poor result does not by
  itself establish that learned premise subsets are worse than SInE subsets.
* Ordinary Vampire proved all 34 Torok successes. Adding Torok to SInE lost
  many proofs that SInE alone obtained within a second, so startup overhead
  alone does not explain the regression. Queue changes and memory costs matter.
* The 20-problem old-training-partition comparison does not prove memorization:
  belonging to a training partition does not establish that every problem
  contributed a successful training example.

A related side experiment in Connections downloaded three HPC-trained models
and evaluated a reproducible tenth within old scheme-4 part 9. This was local
Connections proof search, not Vampire and not an HPC proof comparison:

| Model | Training corpus | TPTP /325 | Mizar /509 | Combined /834 |
| --- | --- | ---: | ---: | ---: |
| Torok | TPTP | 41 | 66 | 107 |
| Kepce | Mizar | 31 | 78 | 109 |
| Osszet | TPTP + Mizar | 43 | 79 | 122 |

All models encountered the same 116 unsupported TPTP inputs and none on Mizar.
The sample differs from the Vampire comparison. Results remain under
`~/TP/connections/results/part9-tenth-20261021/` and
`results/{Torok,Kepce,Osszet}/{TPTP,Mizar,Combined}/part9-tenth-20261021.csv`.
Do not compare these totals directly with Vampire's different input lists.

## Prover9 integration and schedule development, 23 October

Session: `01a16907-75ca-7442-9bd9-2268a3369e39`.

The initial status test used 32 CNF/FOF inputs and obtained six proofs agreeing
with TPTP headers. A negated-quantifier translation issue was first described
as a converter bug, then identified as a known LADR syntax compatibility
limitation: parentheses are required around certain negated formulas. Fixing
the translated PRO001+4 conjecture produced a proof. This was not evidence of
an unsound Prover9 proof calculus. A separate CNF performance pilot proved
64/120 with a 20s CPU/25s wall limit; neither test compares guided search.
The original summaries are copied under `historical/prover9/test-results/`.

The original weighted-queue pilot scored 38/44 against baseline 61 on 122
usable held-out inputs. Uniform queue weights scored 40/43. This motivated
retaining more of the prover's existing selection behaviour.

The following exploratory screen used 61 scoreable problems. These figures
predate the corrected first-refutation stopping rule:

| Strategy | Budget | Recorded proofs /61 |
| --- | --- | ---: |
| All axioms, ordinary order | 20s | 33 |
| All axioms, descending-score order | 20s | 32 |
| Top-16 / top-64 filtering | 8s | 27 / 29 |
| Top-quarter / top-half filtering, minimum eight | 8s | 27 / 27 |
| Score-weighted quarter subset | 8s | 22 |
| Probability thresholds 0.01 / 0.1 / 0.5 | 20s | 18 / 14 / 7 |
| Soft clause weighting | 20s | 29 |
| Neural tie-breaking | 20s | 28 |
| `auto2` | 20s | 29 |
| `auto2` with filtering / soft weighting | 20s | 31 / 31 |

The unequal 8s and 20s allowances prevent a fair direct filtering comparison.
A larger 178-input, 20s filtering run gave baseline 99, probe-plus-filter 97,
direct-filter 96 and random-filter 95.

The session discovered that Prover9 may print one complete refutation and then
continue seeking more proofs for multiple denials. The harness had waited for
the final `THEOREM PROVED` marker. A C runner was added to stop both arms after
the first complete refutation. Historical final-marker results must remain
separate from this later protocol. A corrected 61-input, 20s screen gave
baseline 34, support-16 34, support-64 33 and `auto2` with filtering 32.

The subsequent support-size screen used cached scores with measured scoring
cost charged. At 30s, baseline scored 35/61, and top-1/4/8/16 support scored
34/33/35/34. One apparent top-8 gain followed the unchanged baseline path.
The exact later CLI support1/4/8 variants conservatively leave inputs with at
most 16 axioms alone, so their behaviour differs from that screen; no completed
full comparison for those exact variants was documented.

The main 178-input comparisons then progressed through deterministic top-16,
weighted support, half-budget support, resumed support-16, and ResumeSupport1.
The final result was 106 versus baseline 101 and top-16 103. The discussion
briefly called the best method "top-8"; the actual retained winner is top-1
support with pause/resume. Use the configuration and manifest, not that label.

HPC preparation removed a dependency on the external Connections API, bundled
the required parser/graph code, implemented family-hash job partitions,
changed defaults to 60s and all allocated CPUs, and added resumable output.
Those defaults are not appropriate for local concurrency: pass 18 workers.
No authorization to pull or submit is inherited from these historical requests.

## Prover9 result consolidation, 26 October

Session: `01a17853-5214-7720-9bb4-92346c6d1feb`.

The user supplied `hpc-three-way-60s.tar.gz` and requested a compact CSV without
deleting the archive. The saved merged CSV contains all three strategies;
new runner outputs use a separate CSV per strategy per part. The original
archive remains the detailed source of stage attribution and proof logs.

The final schema uses `problem` with a `.p` suffix, `part`, `declared_status`,
`strategy`, outcome/timing/search statistics, and error/scoring fields. Removed
fields included embedded metadata/details, per-row budgets, CPU, replicate,
and duplicate elapsed seconds. Statistics can be absent after killed searches;
`steps` counts observed given-clause work, not proof length.

The HPC results were later checked against all 18,807 archived proof flags.
The CSV alone does not establish scheduler terminal state or independently
certify proof correctness. Preserve the original archive when analysing it.

## Vampire schedule search and local cross-prover audit, 26 October

Session: `01a177fa-40b8-76d0-9195-eb7ee30c69fa`.

The user requested short local proof-of-concept searches, motivated by the
Prover9 HPC net gain of 96. Native seed, memory, source and model identity were
recorded. Final batches used 18 single-core workers and one shared 60s deadline
per strategy. The native Vampire changes were uncommitted in these experiments;
saved source snapshots and diffs matter alongside checkout HEAD.

Early Otter/support, LRS/soft and filtering candidates were followed by corrected
late schedules. Pausing LRS had distorted throughput estimates and pruning;
the correction uses active process CPU time after resume while retaining an
external wall deadline. Guided attempts also receive their actual internal
slice deadline. Ordinary unpaused search keeps its original clock. The first
two batches predate these corrections and must not be pooled with later runs.

Additional diagnostics:

* A 24-input hard screen raised the scoring cap to 1,024 clauses: SInE proved
  three, filter-16 two, filter-64 one, support-1 one, support-16 two, and SInE
  then support-1 two. Scoring/memory failures supported retaining the 256 cap.
* A second screen at the 256 cap gave SInE-plus-filter one, AVATAR-off filter
  three, soft-1 two, and support-1/16 with ordinary background literal selection
  two each. These are selected difficult cases, not a validation sample.
* GRP186-2 was proved once by support1-soson, selecting one of 18 axioms in
  21.16s. Controlled repeats were cancelled before starting when the user
  requested the last batch. It remains unconfirmed.
* COL046-1 filtering keeps all three axioms; ordinary Vampire proves it in
  2.67s and ordinary AVATAR-off in about two seconds. It is not a model gain.
* A six-input stage-order diagnostic gave six proofs each for ordinary fallback,
  learned-filter-first and learned-support-first; shuffled-support-first gave
  four. The selected sample and timing differences do not replace validation.

The SInE failure-recovery and final AVATAR schedules, the Prover9 HPC cross-check,
and the final local comparison are fully tabulated in the consolidated README
and preserved SINE_FALLBACK snapshot. The user then requested ordinary-first
Prover9 schedules, producing the 50+10 and 40+20 follow-up.

The latest local batches finished with ordinary Prover9 174/300,
ResumeSupport1 177, late-10 176, late-20 177, and Vampire final 212. Audits checked
900 initial plus 600 late outcomes against complete-proof markers and found
no changed source/executable hashes. The recorded experiment processes exited;
this is a historical local completion statement, not a current HPC status.

The final session exchanges correct the earlier "best Vampire schedule" claim:

* All 83 latest held-out Vampire proofs came from ordinary search; neither
  fallback produced a held-out proof.
* Standalone ordinary Vampire was not tested on that same 120-input sample.
* The final schedule and shuffled rankings both scored 129/180 with identical
  proof sets on development.
* Earlier counts of 86/120 belong to a different held-out sample and cannot
  be compared with the latest 83.
* Prover9's HPC 387 versus 372 on checkpoint-held-out inputs is positive,
  but without random support it does not isolate learned ranking from changing
  the initial support.

Accordingly, ordinary Vampire remains the reference configuration. Treat the
Torok stage as experimental. Prover9 ResumeSupport1 is the stronger candidate
for controlled replication; the next experiment should include random support.
