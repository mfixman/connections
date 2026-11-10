# Handoff: axiom-selection goal, Friday 2026-11-06 to Monday 2026-11-09

Written for an LLM analyst. Completeness over readability. All times UTC (UK time = UTC in November).
Primary sources are copied into this folder; the live copies are on the HPC under ~/axiom-goal (being moved to
/rds/user/msf43/hpc-work/axiom-goal by the user on 2026-11-09) and locally under /home/debian/axiom-goal.

Folder map:
- HANDOFF.md (this file): narrative, decisions, models, jobs, changes, issues, pending work.
- goal-files/RESULTS.md: every result table, good and bad, in chronological order, with a verdict and addendum.
- goal-files/PREREGISTRATION.md: every pre-test decision and every test-set look, in order.
- goal-files/PAPER_NOTES.md: paper-oriented synthesis (framing, methods, tables, related work, threats).
- goal-files/REPRODUCIBILITY.md: commits, binary and model hashes, verbatim job commands, data lists.
- goal-files/PLAN.md: running plan, status snapshots, paused items.
- goal-files/LITERATURE.md, LITERATURE_2024_2026.md: literature notes (papers in connections/papers/).
- analysis/: STATS.md (statistics), FAILURES.md (failure analysis), PARETO.md, PORTFOLIO_CI.md, scripts, figures.
- goal-files/*.py and goal-files/p9cores/: experiment scripts (also consolidated in
  packages/axiom_prediction/experiments/axiom_goal/, commit d35fbfc).
- monitor.py, intr_queue.py, submit_once.py, clean_sweep.py, p9_intr.py, hourly.sh: the VPS-side automation.
- monitor-logs/: monitor.log, intr_queue.log, clean_sweep.log, state.json, clean_sweep.json.
- receipts.tsv: every HPC submission since 2026-11-06 00:00 (started_utc, operation, job id or UNCERTAIN, command).
- sacct.psv: Slurm accounting for every job id in receipts.tsv (state, start, elapsed, CPUs, node, exit code).
- CODE_CHANGES.md: git log with stats since 2026-11-06 for Connections, Prover9, SATCoP and Vampire clones.

## 0. Setting

Project: Connections (/home/debian/connections, HPC /home/msf43/connections, branch axiom_predictor), package
packages/axiom_prediction. Task: learned axiom (premise) selection. "Sharbaraz" models: ExtraLargeFull GNNs
(hidden 256, 5 message rounds, 4 layers) trained on SATCoP unsat-core membership of axiom clauses, collected by
running native SATCoP (Rust, binary satcop-e66dfc8) with policies SatCoP and SatResetCoP over TPTP problems with
seeds 0-19; model name SharbarazN = trained on seeds 0..N-1. Split: ProblemSplit scheme 5, --split 10; parts 0-7
train, part 8 validation (1,426 refutable problems; 608 known-unsat CNF for Prover9/E/SPASS/iProver), part 9 test
(1,390 refutable; 581 CNF). Budget 60 s per problem unless stated. Metrics: proved .p files (primary), TPTP
families (leading [A-Z]{3}\d{3}; solved if any member proved), paired gains/losses, exact two-sided McNemar.

User goals in order:
1. Friday (/goal): NN-guided axiom selection improving SATCoP and Prover9 by >= 5%, or Vampire >= 2%, test set,
   simple, not p-hacked, 80/10/10 split; notify on final result or daily significant improvement.
2. Saturday evening (/goal rewritten): ONE model, +10% on BOTH SATCoP and Prover9 or +5% Vampire, per file and
   per family, controls (random/shuffled, SInE, symbol overlap) in identical schedules, stage attribution,
   preregister on part 8, run part 9 once, 3 seeds + McNemar, AUC/recall@k, transfer framing, reproducibility.
3. Monday: "keep going", then at 10:45 UK "make sure everything that can be evaluated is evaluated"; then
   parallel programme (more provers, finish collections, publication experiments, simplicity/performance Pareto,
   keep improving; Monday not final). User clarified: headline metric per file, families alongside; the user
   wants IJCAI (or CADE) publication.

Outcome: goal 1 met (Prover9 +6.3%, SATCoP +6.4% on test). Goal 2 not met: best single model gives SATCoP
+6.4% / Prover9 +6.9% (+7.5% with a two-ranking portfolio); Vampire +2.6% comes from SInE, not the GNN.

## 1. Results (good and bad)

See goal-files/RESULTS.md for every table; this is the index. Unless stated, model = Sharbaraz5 epoch 20
(/home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts), chosen on part 8.

Test set (part 9), headline:
- SATCoP, equal-thirds schedule (plain SatCoP 20 s, SatCoP with model weights 20 s, SatResetCoP with model
  weights 20 s; union of three real time-capped runs): files 635/597 (+6.37%), families 445/418 (+6.46%), seed 0;
  seeds 1,2: 639 (+7.04%), 632 (+5.86%); mean +6.42%; 95% bootstrap CIs exclude zero. Random weights 604/599/599
  (+0.6% mean). SInE weights 622 (+4.19%), families 432 (+3.35%). Symbol-overlap weights: SatCoP stage 536 alone,
  SatResetCoP stage rerunning (cpu2 job 40409886). Plain SatCoP 60 s baseline 597 / 418.
- Prover9 (CNF, 581): baseline 333 / 277. ResumeFilter16Support1 355 (+6.61%) / 290 (+4.69%) [shuffled 346];
  Ladder (1 s probe, top-16/64/256 x 8 s, resume) 348 (+4.50%) [shuffled 332 (-0.30%), overlap 351 (+5.41%), SInE
  339 (+1.80%)]; LadderSupport1 356 (+6.91%) / 289 [shuffled 344, overlap 349, SInE 345]; DualLadder (learned and
  overlap cutoffs alternating) 355 [shuffled-learned 350]; DualLadderSupport1 358 (+7.51%) / 291 (+5.05%)
  [shuffled-learned 354 (+6.31%); dual-vs-shuffled 9/5, p = 0.42]; minimal ablation ResumeFilter16 (one 2 s top-16
  search then resume) 346 (+3.90%), 13/0, p = 0.0002 [shuffled 334, 1/0]; ResumeSupport1 346 (+3.90%).
  Earlier with Sharbaraz1 e100: ResumeFilter16Support1 354 (+6.31%), shuffled 346, overlap 346.
- Vampire (LRS, 1,390): baseline 911; resume-sine:24 935 (+2.63%, p 1.8e-4); learned ladder 914 (+0.33%);
  random ladder 906; SInE then learned 934. GNN adds nothing over SInE.
- E (CNF 581): part 9 running (cpu2 job 40409886). Part 8: baseline 467, E SInE 470, learned ladder 468, shuffled
  466, overlap 466 (near ceiling, neutral).
- SPASS 3.9 (part 8 only): baseline 347, learned ladder 346, shuffled 345 (neutral). iProver 3.9.4 (part 8 only):
  baseline 412, learned ladder 413, shuffled 408 (neutral; learned wins inside the filtered windows 26 vs 4 but the
  25 s ladder loses late baseline proofs).
- Cross-prover portfolio on the 581 shared CNF test problems: plain SatCoP 247, plain Prover9 333, union 380 /
  314 families; guided union (thirds + LadderSupport1) 395 / 321 (+3.95% / +2.23%); control union 388 / 320.

Validation (part 8) facts that drove decisions:
- Clean baselines (after the orphan fix): SatCoP 691-692, SatResetCoP 317-320, Prover9 379 / 288 families.
- Guided SatResetCoP: Sharbaraz5 e20 461-462 (+44%); SInE 338; overlap 366; random 320. Guided SatCoP: 671 (-3%)
  for every checkpoint; weights hurt plain SatCoP.
- Checkpoint sweep (every 10th epoch, 4 models): SatResetCoP peaks at epochs 10-30 (Sharbaraz5 e20 best, 461);
  epoch 100 of Sharbaraz1 only 395. All four peak checkpoints give equal-thirds 722-725 and ceilings 743-749.
- Ceilings (union of plain + every guided stage with its own full 60 s): SATCoP 758 (+9.54%) files / 470 (+9.05%)
  families with Sharbaraz5 e20; Prover9 learned ladders 407 (+7.4%), learned+overlap 415 (+9.5%), all rankings' ladders
  416 (+9.8%), all 12 runs 421 (+11.1%); Vampire all arms 1032 (+7.3%), baseline+learned arms +5.1%, baseline+SInE
  arms +4.9%.
- Ranking quality (618 part-8 problems with both classes; recall over problems with > k axioms): all nine
  checkpoints AUC 0.895-0.908, recall@16 0.64-0.69, recall@64 0.87-0.90. Sharbaraz5 e20: 0.903 / 0.686 / 0.895.
  Union-of-cores labels: 0.900 / 0.662 / 0.881 (no change). Ensemble of S5/S10/S15 e20: 0.916 / 0.697 / 0.906 but
  Prover9 ladder 398 vs 402 (no search gain). LargeFull probe e20: 0.895 / 0.644 / 0.880 (worse). ExtraLargeFull with
  per-problem loss e25: 0.912 / 0.641 / 0.814 (worse recall).
- Negative schedule experiments (part 8): temperature T in {0.5, 2, 4} on guided SatCoP (671 -> 674 / 666; T 0.5 run
  invalid, /tmp full); weight mixing lambda 0.25 / 0.5 (673 / 666 vs 692); top-16 / top-64 filter stages (filters alone
  385 / 590; random filters 185 / 321; adding filter stages to the thirds does not help: 722-714 vs 724); second guided
  SatResetCoP seed as a 4th stage (712); budget variation (shorter budgets reduce the SATCoP gain: +1.8% at 30 s,
  negative at 15 s); SatCoP-then-unguided-SatResetCoP (+1%).
- Prover9 budget sweep (part 8): learned ladder +4.93% (30 s), +6.07% (60 s), +5.70% (120 s); shuffled < 1% at each;
  LadderSupport1 +4.38 / +7.12 / +6.48%.
- Prover9 Pareto front (part 8, learned): ResumeSupport1 394 (= shuffled 394, schedule effect) = ResumeFilter16 394
  (shuffled 383) < RF16S1 401 < Ladder 402 < LadderSupport1 406 < DualLadder 411 < DualLadderSupport1 412.
- Failure analysis (analysis/FAILURES.md): SATCoP never-solved (668/1,426) are equational (31% of never-solved vs
  17% of the set) and small CNF problems (140 with <= 16 clauses) that the prover cannot solve with any selection
  (plain Prover9 proves 128 of SATCoP's 298 never-solved CNF); the 14 thirds losses are large problems (median 39 s of
  plain search needed) where the model is poor (recall@16 0.32, median core 39). Prover9 never-solved (187/229) have
  the core inside the top 16 (recall@16 0.74) and fail on search; 74 have <= 64 clauses; top-256 stage gives 0
  gains; gains concentrate in > 1024-clause problems.
- Vampire SInE artefact: the runner's standalone `sine` mode (tolerance 1, no fallback) scored -33% because
  216/355 lost problems ended "Refutation not found, incomplete strategy"; the fair baseline is SInE as a resume stage.

Earlier/contaminated results (kept, not used): W&B backfill validation curves before the orphan fix (2026-11-06
afternoon) and the first intr random-control runs (~/axiom-goal/satcop-part8/orphan-bug/) are depressed and noisy.
The first Prover9 part-8 comparisons (part8-torok etc.) used 76 workers; later SATCoP runs used 32 workers; each
table compares within one setup.

## 2. Models trained or built

| Model | Trained when / job | Data | Why | Quality (part 8) | Use |
|---|---|---|---|---|---|
| Sharbaraz1 | monitor trainer, 2026-11-06 (job 40134874, 2h06, 100 epochs) | seed 0 (parts 0-7), joined v3 | first of the per-5-seeds series | AUC 0.901-0.902, recall@16 0.66-0.68; SatResetCoP peak e30 436 | Prover9 first test result (e100) |
| Sharbaraz5 | monitor trainer, 2026-11-06 13:36-17:36 (40134887, TIMEOUT; later resumes) | seeds 0-4 | 5-seed model | e20: 0.903 / 0.686 / 0.895; SatResetCoP 461 | THE single model for all headline results |
| Sharbaraz10, Sharbaraz15 | monitor trainers, resumed in 4 h slots | seeds 0-9, 0-14 | 10/15-seed models | e20: 0.906 / 0.686 / 0.897 and 0.908 / 0.677 / 0.900; SatResetCoP 451 / 455 | ceilings equal to S5; not selected |
| Sharbaraz20 | monitor trainer, queued (gpu2) | seeds 0-19 | 20-seed model | pending | none yet |
| Sharbaraz20short | reserve slot long-job-9 repurposed (12 h job), pending since Sat | seeds 0-19, 30 epochs | early-stopped big-data model | pending | none |
| Variant-LargeFull-S5 | 40306928, 1 h, 20 epochs, Sun 18:03-18:37 | seeds 0-4 | smaller model probe | e20 0.895 / 0.644 / 0.880 (worse) | none |
| Variant-XLProblemLoss-S5 | 40300567, 2 h, 25 epochs, Sun | seeds 0-4, --loss-weighting Problem | loss-weighting probe | e25 0.912 / 0.641 / 0.814 (worse recall) | none |
| ensemble3-e20.ts | built Sun (make_ensemble.py) | S5/S10/S15 e20 averaged logits | MizAR-60-style averaged ranking | 0.916 / 0.697 / 0.906; Prover9 ladder 398 vs 402 | none |
| P9cores2 | 40384641 (20 min, Mon 11:00-11:04), 20 epochs | 3,437 Prover9 proof cores (parts 0-8 CNF; p9_cores.py) | own-prover labels vs transferred labels | pending (cpu2 job 40409578) | pending |
| random.ts, overlap.ts, sine | controls (make_random.py: unseeded randn; make_overlap.py: fraction of an axiom's symbols in the conjecture; `sine` = marker file selecting the SInE predictor in Prover9) | | controls | | all control arms |

Training commands: ExtraLargeFull, --epochs 100 (monitor) or 20-30 (probes), --max-example-mb 10, --split 10
--parts 0..7, --evaluate 8 --validation-run-ratio 0 --evaluate-every 1 (added Fri evening; earlier trainers had no
held-out evaluation, so their W&B ROC curves are training-set), --wandb. Dataset dirs results/big-train/datasets/<name>
(data.npz + metadata.json); trainer dirs results/big-train/<name>/model/epoch-XXXX.ts.

## 3. Jobs

receipts.tsv lists every submission made through the shared lock (monitor, queues, agents) since 2026-11-06 00:00:
390 rows; sacct.psv gives their Slurm states. Categories: collection SatCoP-sN-pM / SatResetCoP-sN-pM (seeds 10-19,
200 jobs, all COMPLETED by Sat 20:08); validate-<model>-wN (W&B backfill, 32 CPUs, 12 h, nice 5 -> 7 -> 15); clean-
<model>-eN (clean sweep, every 10th epoch, 32 CPUs); Sharbaraz<N> trainers (gpu2, 4 h); long-job-9..12 (reserve
GPU slots, 12 h, 4 GPUs, 128 CPUs; slot 9 repurposed); p8-*/p8w32-*/t9-*/t20-*/t20sN-* (SATCoP comparisons via
run_selection.py: 76 or 32 workers); p9-part8-*/p9-part9-* (Prover9 compare.py); vampire-*; e-*/e8-*/e9-*;
spass8-*/iprover8-*; quality-*/model-quality-* (ranking metrics); p9cores*/P9cores2-* (Prover9-core pipeline);
Variant-* (probes); cpu2-eval-* (final cpu2 jobs). Approximately 135 jobs on Fri-Sat (24 h), of which 43 on intr.
Known failures: satcop-part8-random (network_size bug), p9-gain-stages x2 (/tmp full; predictor reuse), p9cores-*
first collection (variable-class bug: 44 proved), p9-part8-simple (I/O stall, requeued), model-quality-union x2
(frozen dataclass; variable shadowing), e-part8 ladders (1 h limit; made resumable), T0.5 run (551 /tmp errors), two
validation OUT_OF_MEMORY, several "Uncertain submission" (sbatch gave no id: slow login node and, on Monday, home
quota blocking icelake-batch's batch-script write under ~/tmp).

## 4. Design and experiment decisions (chronological; details in PREREGISTRATION.md)

1. Fri ~16:00: preregistration: scheme 5 split, 60 s equal budget incl. scoring, relative gain; SATCoP model by mean
   validation over epochs 81-100 then epoch 100; Prover9 ResumeSupport1 with the epoch-100 checkpoint with most
   part-8 proofs; one part-9 run with baseline, guided and random control.
2. Fri: user prefers `run` accepting plain .ts (random.ts) -> commits 4db793c, c0f347c. Single model preference
   for Sharbaraz over Torok.
3. Fri 17:30: orphaned-SATCoP bug found (about 1,500 leftover provers, 85-134 per node): earlier validation
   contaminated. Fix 43c0fff; amendment: all comparisons on clean code, one arm per job, model chosen by clean
   part-8 runs of epoch-100 checkpoints.
4. Fri evening: user grants HPC pull permission (via GitHub only), asks Pushbullet for any permission question,
   says a bit of test-set reuse for small decisions is fine; large decisions on validation.
5. Fri 23:00: clean sweep (every 10th epoch) instead of full backfill rerun; choose epoch by best clean
   SatResetCoP; part 9 once. Prover9 part-8 variants: ResumeFilter16Support1 chosen (401) over ResumeSupport1 (392).
6. Sat 01:10: Prover9 part-9 run with Sharbaraz1 e100 (+6.31%); symbol-overlap control added afterwards (an
   additional test look, logged).
7. Sat 10:00: decision moved earlier than planned (sweep complete, trainers queued): Sharbaraz5 e20 chosen.
   Part-9 SATCoP runs: SatResetCoP baseline / guided / random, SatCoP baseline (32 workers).
8. Sat 12:30-19:00: guidance helps SatResetCoP but not the strongest configuration (SatCoP); weight mixing tried
   (user approved) -> negative; equal-thirds schedule fixed as the untuned design; part 8 +4.62% missed the +5% gate,
   part 9 run anyway (disclosed); confirmed with real 20 s stage runs (+6.37%); 3 seeds later.
9. Sat 19:30: user requires a single checkpoint for all provers -> Prover9 rerun with Sharbaraz5 e20 (+6.61%).
10. Sat 20:45 (Fable session): goal 2 design: Prover9 ResumeLadder 16/64/256 x 8 s (DeepMath/MizAR cutoffs);
    SATCoP --filter-top; controls shuffled, SInE (predictor added to Prover9), overlap; checkpoint choice by combined
    gain over shuffled; part 9 once with real stage runs; reserve slot 9 repurposed to Sharbaraz20short.
11. Sun: SInE control results; Vampire fair baseline = resume-sine:24; two new-model probes (LargeFull 1 h, XL
    per-problem loss 2 h; shorter jobs start sooner); ensemble probe; budget sweep; failure analysis -> dual-ranking
    ladder (GNN + overlap) tested on part 8 (411) then part 9 once (355 / 358).
12. Sun-Mon: user asks for per-file headline, more provers (E, SPASS, iProver built from source), Pareto analysis,
    "evaluate everything"; Prover9-core labels collected (p9_cores.py) and model trained (P9cores2); part-9
    completeness runs (SATCoP SInE/overlap controls, E, budgets).
13. Mon 11:45: user takes over intr; home quota at hard limit; remaining evaluations moved to three cpu2 jobs
    writing to /rds/user/msf43/hpc-work/axiom-goal-cpu2/.

Other standing decisions: notifications only for positive breakthroughs / near goal / fatal / permission / token
(Sat); families reported alongside files (Sat), files as headline (Sun); no --intr for cpu2-sized work; at most one
intr job at a time via intr_queue.py; 60 s between *-batch calls via submit_once.

## 5. Changes to code and directories

Connections (branch axiom_predictor; see CODE_CHANGES.md): 4db793c (plain .ts in run), c0f347c (network_size for
plain .ts), 43c0fff (PR_SET_PDEATHSIG kills native SATCoP with its worker; fixes orphans), 9c7c588 (run
--temperature), 5a4816d (per-worker scratch dir deleted by the supervisor; fixes /tmp fill), 0a7148b and earlier
(backfill script options, by the user/me before Friday), d35fbfc (experiments/axiom_goal consolidated scripts +
README; force-added into the gitignored experiments/). Uncommitted: experiments/mizar_monitor.py (pre-existing),
papers/download_manifest.json + 16 new papers (two agent batches), handoff folder.
Prover9 (~/TP/Prover9, branch axiom_selection): a27a68d (pidfd fallback, Fri), e7cb3e1 ResumeLadder, 1ab1e6e SInE
predictor (model path named `sine`), 98db53d ResumeLadderSupport1/LongLadder, ce26c08 + 4a70421 dual-ranking ladder
and second-predictor thread fix. Note: ladder 'ranking' per stage is recorded in memory but not exported to CSV.
SATCoP (~/TP/satcop, axiom_selection): 0ecdb06 --weight-mix, 177002a --filter-top; binaries satcop-e66dfc8 (all
main results), satcop-0ecdb06, satcop-177002a under /home/msf43/validation-sharbaraz/ with sh wrappers
satcop-0ecdb06-mix0.25/0.5 and satcop-177002a-top16/64.
Vampire (~/TP/vampire, axiom_selection): 0270a14 compare.py --model.
New provers on HPC: ~/TP/eprover (E 3.5.5, bin/eprover), ~/TP/SPASS (3.9), ~/TP/iprover (3.9.4; OCaml via
modules; ~/.opam symlinked to /rds/user/msf43/hpc-work/opam). Runners: axiom-goal/e_compare.py, spass_compare.py,
iprover_compare.py (resume-ladder design with SIGSTOP/SIGCONT, resumable CSVs).
VPS automation (~/sharbaraz-monitor): monitor.py (collection, joins, trainers, validation; validation nice 15;
trainer template now has --evaluate 8; SSH timeouts 420/240 s; hpc_answers() check before token alerts),
intr_queue.py (serial intr submissions; sacct terminal-state wait; stalled-poll logging; name-based reconciliation
of lost submissions; 300 s ssh timeout), submit_once.py, clean_sweep.py (every-10th-epoch clean validation into
results/big-train/validation-clean-<model>/), p9_intr.py, hourly.sh (stopped Sat).
HPC files: ~/long_job_9.py rewritten (Sharbaraz20short trainer; placeholder saved as .reserve-placeholder);
~/long_job_6/7/8.py given --evaluate 8; ~/axiom-goal/ (all experiment inputs/outputs; moving to RDS);
~/axiom-goal/manifests/ (part8/part9-cnf-unsat.json, parts0-8-cnf-unsat.json, vampire-part8/9.json);
part8/part9-selection.json (refutable FOF+CNF lists); ~/tmp/connections-4db793c (temporary checkout, deleted Sat);
~/tmp finished job logs partially archived to /rds/user/msf43/hpc-work/tmp-archive (404 dirs); scratch dir
~/axiom-goal/scratch. Deleted (with notice): the contaminated 44-example P9cores dataset and shard; junk from the
first collection attempt. AGENTS.md: split rule, 4-minute job rule, RDS slowness, GPU long-job slots section.
Memories (VPS ~/.claude/.../memory): intr jobs unlimited, goal notifications, permission requests notify, HPC pull
permission (GitHub only), test-set usage, family metric (files headline), single model.

## 6. Known issues and operational findings

- Orphaned SATCoP processes (fixed 43c0fff) and worker temp files filling compute-node tmpfs (fixed 5a4816d);
  both silently lowered proof counts in earlier runs.
- Login node slowness (Sun night): ssh round trips up to a minute; sacct can report RUNNING just after squeue drops a
  job; sbatch sometimes returns no id ("Uncertain submission"); the queue's "is intr free" poll returned empty for
  ~5 h (Mon 04:31-09:52) and nothing was submitted; fixes in intr_queue.py as above.
- Home quota: hard limit 57.7 GB reached Mon ~10:00 (OCaml build). icelake-batch writes batch scripts to ~/tmp, so
  submissions fail when home is full. User is handling storage separately (job move-axiom-goal running at 12:00).
- Monitor false alarms "token may be needed" on slow nights (twice); mitigated with longer timeouts and a quick
  login probe before alerting.
- random.ts is unseeded (controls not bit-reproducible; 3 seeds show stability).
- Prover9 compare.py export drops the per-stage 'ranking' key; attribution by time windows or stages.py reruns.
- The schedules are unions of separately time-capped runs (same total budget), not one process.
- Part-9 looks (all logged): Prover9 (S1 e100 RF16S1 + controls; overlap; S5 e20 RF16S1; ladders; dual; simple
  arms; budgets pending), SATCoP (thirds + random + SatCoP baseline; 3 seeds; SInE/overlap controls), Vampire (one
  run), E (running). The +5% part-8 files gate for the thirds was missed (+4.62%) and part 9 run anyway.

## 7. Pending at handoff (2026-11-09 12:15 UTC)

- cpu2 jobs 40409578 (P9cores2 quality + Prover9 part-8 schedules), 40409886 (E part 9 + overlap SatResetCoP
  stage), 40410154 (Prover9 part 9 at 30/120 s): outputs in /rds/user/msf43/hpc-work/axiom-goal-cpu2/.
- GPU: Sharbaraz20 trainer and reserve slots long-job-9..12 (slot 9 = Sharbaraz20short) pending on gpu2.
- Monitor and clean_sweep.py still running on the VPS; hourly digest stopped.
- Not done: REPRODUCIBILITY.md lacks the E/SPASS/iProver/P9cores entries; W&B backfill curves before the fix are
  contaminated (clean numbers only every 10th epoch); Vampire test for E/SPASS/iProver schedules beyond part 8.

## 8. Late additions (2026-11-09 12:40 UTC)
- The first three cpu2 evaluation jobs (40409578, 40409886, 40410154) failed in seconds: `$S` in the inline
  `bash -c` command was expanded to empty by the quoting layers (ssh -> icelake-batch -> batch script), so paths
  became `/model_quality.py`. Resubmitted as script files under /rds/user/msf43/hpc-work/axiom-goal-cpu2/jobs/
  (eval_p9cores2.sh, eval_e9_overlap.sh, eval_p9_budget.sh; common.sh sets S/C/B/O/TP): jobs 40411859, 40412070,
  40412287. Lesson: never pass multi-variable shell snippets through the batch wrappers; write a script and run it.
- JOBS.md: every submission since Friday with Slurm state; receipts.tsv and sacct.psv are the raw inputs.

## 9. Final additions (2026-11-09 14:05 UTC)
All pending evaluations from section 7 completed (see RESULTS.md, last sections): P9cores2 own-label model does not
beat the transferred model (Prover9 part 8 LadderSupport1 405 vs 406; worse ranking of Prover9's own cores: recall@16
0.680 vs 0.737); E part 9 neutral (422/423/422/422); Prover9 part-9 budget sweep flat (+4.0% at 30 s, +5.3% at
120 s, shuffled <= +0.6%); SATCoP overlap thirds part 9 +2.35% (SInE +4.19%, learned +6.37%, random +1.17%).
Outputs: /rds/user/msf43/hpc-work/axiom-goal-cpu2/. Nothing is queued on the HPC; monitor.py and clean_sweep.py run
on the VPS; GPU jobs Sharbaraz20 and long-job-9..12 remain pending.

2026-11-10: the ~/TP prover clones moved into ~/connections/external (submodules; symlinks removed), so every
`~/TP/<prover>` path in this handoff now means `~/connections/external/<prover>`.
