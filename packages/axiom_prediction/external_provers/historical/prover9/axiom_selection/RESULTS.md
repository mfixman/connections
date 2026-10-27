> Historical result: this pilot used Prover9's final `THEOREM PROVED` message.
> Later development found that automatic handling of multiple Horn denials can
> produce a valid first refutation and then continue searching. These figures
> therefore measure that older stopping rule, not consistently time to first
> refutation. New experiments use `native_search.c` to stop at the first complete
> printed proof for both arms. Do not combine the two sets of measurements.

# Prover9 / Torok pilot, 23 October 2026

**This integration did not improve overall performance.** Baseline Prover9
proved 61 of 122 usable held-out problems. Torok proved 38 and 44 with seeds 1
and 2. Neither seed proved anything the baseline failed to prove. There was
one faster common proof, detailed below; that does not establish a generally
better problem subset.

| Configuration | Proved / 122 | Proved by 5 s | Median wall time among its successes | PAR-2 mean |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 61 (50.0%) | 55 | 0.82 s | 15.96 s |
| Torok, seed 1 | 38 (31.1%) | 18 | 5.06 s | 22.34 s |
| Torok, seed 2 | 44 (36.1%) | 21 | 5.07 s | 21.17 s |
| Uniform, seed 1 | 40 (32.8%) | 36 | 0.77 s | 20.77 s |
| Uniform, seed 2 | 43 (35.2%) | 36 | 0.77 s | 20.17 s |

Times include process startup, shared CNF parsing, model loading, graph
construction, inference, and proof search. PAR-2 charges 30 seconds to an
unsolved run under the 15-second cap; lower is better. Success-only medians
refer to different success sets, so they are not matched speedup estimates.
On common successes, Torok's median end-to-end time ratios to baseline were
6.81 and 6.74, respectively.

Torok lost 23 baseline proofs in seed 1 and 17 in seed 2; 17 losses occurred in
both seeds. Uniform guidance also lost 21 and 18 baseline proofs, respectively.
Thus neural startup is not the whole explanation: the four-weighted-to-one-native
schedule also underperforms without any model inference. This comparison tests
one integration heuristic; it does not establish that the network is useless
with a different integration or prover.

## Individual improvement

For `SWC059-1` (198 clauses), all measurements were on the same assigned CPU:

| Configuration | End-to-end time | Prover search CPU time |
| --- | ---: | ---: |
| Baseline | 14.24 s | 13.32 s |
| Torok, seed 1 | 13.75 s | 7.52 s |
| Torok, seed 2 | 8.30 s | 3.13 s |
| Uniform, either seed | timeout at 15 s | unavailable |

The two Torok times average 11.03 seconds, an observed 1.29x end-to-end speedup
(22.6% less time). This is an exploratory individual example, identified after
the run, with substantial seed variation. It is not a pre-specified subset or
strong statistical evidence. No baseline failure was solved by Torok in this
sample, so there are no additional-proof gains to report.

## Protocol and accounting

The main pilot lasted **397.81 seconds (6 min 38 s)**. It ran 124 matched groups,
620 process invocations, with at most 18 single-core workers inside a 19-logical-CPU
affinity set on an Intel Core i7-1370P. Each group contained the baseline once
and Torok/Uniform at seeds 1 and 2, in a deterministic randomized order. Every
arm had the same strict 15-second whole-process wall budget and 512 MB Prover9
memory setting. PyTorch inference used one thread. The model and Python process
were cold-started for each Torok arm; warm-service performance is not measured.

Candidates were known-unsatisfiable TPTP CNF problems in scheme-4 part 9 of 10,
matching the checkpoint's recorded training split. One random variant per
six-character TPTP family was selected, using seed 2026102303. The candidate
pool contained 462 families. Groups were launched from that randomized order
for 330 seconds; all started groups were completed. The usable sample covers
24 domains. There was no filtering on baseline success, size, or TPTP rating,
and no tuning of the mixture or seed using this test's results.

Two groups (`SYO678-1`, `SYO665-1`) were rejected in all arms because this first
adapter does not serialize their quoted uppercase symbols (`E`, `LE`). They
are reported as common input errors and excluded from the 122 usable groups.
`SWV838-1` hit Prover9's existing `renum_vars_recurse: too many variables` error
under Uniform seed 1. It remains in the denominator as an unsuccessful run;
the other arms also failed to prove it within their budgets.

All emitted native inputs were checked to be byte-identical across arms for
each usable group. The baseline loads no neural model. Torok and Uniform retain
all clauses and differ only in their weights and the cost of producing them.
Both use the same four-to-one weighted/native schedule. The inherited-parent
weight rule and that mixture are fixed experimental heuristics, not trained
network behavior. Prover9's existing pruning and resource limits still apply.

The upstream objects use their default `-O -Wall`; the enhanced given-selection
and adapter objects use `-O2 -Wall`. This pilot therefore is not a perfectly
compiler-matched microbenchmark, and small timing differences should not be
attributed solely to the neural scores. Torok and Uniform use the same binary.
The manifest records both executable hashes and flags. No inference-code source
changes were made.

## Validation and artifacts

Validation passed for the weighted sampler's empirical frequencies, removal and
capacity growth, geometric parent inheritance, copied TorchScript/checkpoint
probability parity, CNF literal signs, inequality, truth constants, native equality,
and disabled-adapter proof/search parity. These checks do not independently
certify every generated proof.

- [Summary JSON](results/pilot-20261023/summary.json)
- [All paired results](results/pilot-20261023/results.json)
- [Frozen sample, budgets, hashes and revisions](results/pilot-20261023/manifest.json)
- [Model provenance](models/provenance.json)
- [Runner and reproduction instructions](README.md)

Raw native inputs, driver logs, and available proof logs are stored in each
problem's result directory. These bulky results and build files are locally
retained but git-ignored. `analyse.py` regenerates the summary from the raw JSON.
The ordinary Prover9 executable and its default behavior remain available.
