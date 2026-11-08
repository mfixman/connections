# Axiom-selection goal: experiment scripts

Cleaned copies of the ad-hoc scripts that produced the axiom-selection results (SATCoP, Prover9, Vampire and E on
TPTP v9.0.0, `ProblemSplit(split = 10, scheme = 5)`: parts 0-7 train, part 8 validates, part 9 is the test set).
The HPC copies in `/home/msf43/axiom-goal/` were the source; behaviour is unchanged, but every `/home/msf43` path is
now a command-line option whose default is the original location (`paths.py`). The scripts are plain
`python script.py ...` programs run from any directory; they import `paths.py` and `results.py` from their own folder.
`make_selection.py` and `model_quality.py` need a Connections checkout (`--connections`), `stages.py` and `e_compare.py`
need the Prover9 repository's `axiom_selection/` directory (`--prover9`), and the three `make_*.py` scripts need torch.

## Shared helpers

`paths.py` holds the default locations: the goal directory (`/home/msf43/axiom-goal`), the Connections checkout, the
TPTP root (`/rds/user/msf43/hpc-work/TPTP`), `results/big-train` for checkpoints, Prover9's `axiom_selection/` and the
E binary. `results.py` reads both result CSV formats (SATCoP writes `proved` as `true`/`false` and `error` as `false`
or a message; the Prover9 and E runners write `True`/`False` and an empty `error`), and provides the TPTP family of a
problem (`SYN003` for `SYN003-1.p`), percentage changes and the exact two-sided McNemar test on discordant pairs.

## Scripts

**`run_selection.py`** runs `axiom_predictor.py run` on the problems of a selection file, so every arm sees exactly the
problems of the validation backfill. It `execv`s the interpreter that runs it, with
`-u <checkout>/axiom_predictor.py run <remaining arguments> <problems...>`.
`python run_selection.py part9-selection.json /home/msf43/connections --policy SatCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output satcop-part9/baseline-SatCoP.csv`

**`make_selection.py`** writes the refutable TPTP problems (FOF and CNF) of one split-10 part, as
`axiom_prediction.validation_run.refutable` selects them, to a JSON file with `held_out`, `problems` and `split`.
`python make_selection.py 9 part9-selection.json` (options `--connections`, `--tptp`).

**`make_random.py`** exports `RandomScores`, a TorchScript module with the Sharbaraz interface
`(nodes, edges, axioms, conjectures) -> logits` that returns `torch.randn` per axiom: the random-ranking control.
It is unseeded, so random-control runs are not bit-reproducible. `python make_random.py models/random.ts`

**`make_overlap.py`** exports `SymbolOverlap`, the untrained control that scores an axiom by the fraction of its symbols
that also occur in the conjectures (walking atoms down to every subterm through the graph edges), returned as logits.
`python make_overlap.py models/overlap.ts`

**`make_ensemble.py`** exports one TorchScript module averaging the logits of several checkpoints (MizAR 60-style
averaged ranking); this experiment is not part of RESULTS.md.
`python make_ensemble.py models/ensemble3-e20.ts a/epoch-0020.ts b/epoch-0020.ts c/epoch-0020.ts`

**`families.py`** prints, for each result CSV, the proved files and proved TPTP families with their change against the
first CSV, plus the families newly proved and lost. `python families.py baseline-SatCoP.csv sharbaraz5-e20-SatCoP.csv random-SatCoP.csv`

**`compare_arms.py`** is the paired comparison of strategy CSVs against a baseline CSV on their common problems:
proved counts, gains, losses and error rows. `python compare_arms.py part9-test/results-Baseline.csv part9-test/results-ResumeFilter16Support1.csv`

**`schedule.py`** simulates a two-stage schedule from per-problem proof times: the first CSV's run for `t` seconds, then a
second run for the rest of the budget (`--total-seconds`, default 60), for every `t` in a comma-separated list and every
second-stage CSV. `python schedule.py satcop-part8-w32/baseline-SatCoP.csv 20,30,40 satcop-part8-w32/sharbaraz5-e20-SatResetCoP.csv`

**`thirds.py`** simulates the equal-thirds schedule (SatCoP, then SatCoP with weights, then SatResetCoP with weights)
from stage CSVs in one folder: a problem counts when any stage proves it within `--stage-seconds` (default 20), for the
learned (`--learned-name`, default `sharbaraz5-e20`) and random stages; also reports the stage runs' error counts.
`python thirds.py satcop-part9-t20 baseline-SatCoP.csv`

**`seeds.py`** builds the three-seed part-9 table: for each seed, the union of the real 20 s stage runs (baseline SatCoP,
learned or random SatCoP, learned or random SatResetCoP) against plain 60 s SatCoP, with files, families, paired
gains/losses and McNemar p, then the per-arm mean and range. `python seeds.py --stages satcop-part9-t20 --plain satcop-part9/baseline-SatCoP.csv`

**`designs.py`** is the equal-split design grid on part 8: for each candidate stage list (guided SatCoP, guided
SatResetCoP, top-16 and top-64 filters) the budget is split equally between the baseline and the stages, and files and
families are counted for learned and random stages from per-problem proof times. `python designs.py satcop-part8-w32`

**`stages.py`** reruns the Prover9 problems that a strategy gained over `results-Baseline.csv` with stage recording,
and writes which stage (top-16 filter, top-1 support, plain Prover9) proved each one to `gain-stages.json` in the run
directory. It drives Prover9's `premise_run.solve` in a spawn pool. `python stages.py --run prover9/part9-test --model /home/msf43/connections/results/big-train/Sharbaraz1/model/epoch-0100.ts --workers 32`

**`p9table.py`** prints the Prover9 part-9 table (ladder and LadderSupport1 with learned, shuffled, overlap and SInE
rankings, plus the earlier ResumeFilter16Support1 run): files, families, paired gains/losses and McNemar p against the
`part9-sharbaraz5e20` baseline, then learned-vs-shuffled, overlap and SInE discordant counts. `python p9table.py --prover9 prover9`

**`model_quality.py`** measures ranking quality on the part-8 examples of a joined dataset: per-problem ROC AUC
(Mann-Whitney with tied ranks) and recall@16/64 of unsat-core axioms, per checkpoint, written as JSON. `--union`
scores against the union of cores over all seeds of a problem and appends `-union` to the output name.
`python model_quality.py --threads 32 results/big-train/Sharbaraz5/joined results/big-train/Sharbaraz5/model/epoch-00{10,20,30}.ts --output model-quality-part8.json`

**`e_compare.py`** runs the E prover comparison on a CNF manifest: `Baseline`, `ESInE` (`--sine=Auto`) and the resume
ladder (`Ladder`, `ShuffledLadder`): a 1 s probe, SIGSTOP, bounded scoring with the Prover9 predictor, filtered E runs on
the top 16, 64 and 256 axioms for 8 s each, then SIGCONT of the paused baseline. Rows are appended as they finish, so a
timed-out job resumes. See the command list below for the exact invocation.

**`sweep_table.py`** prints the checkpoint sweep: clean part-8 validation results per model and epoch from
`validation-clean-*/results-e*.csv`, with changes against the recorded baseline. `python sweep_table.py --results /home/msf43/connections/results/big-train`

## Commands behind the test-set results

Copied from the local submission receipts (REPRODUCIBILITY.md, section 6, collected 2026-11-08). Every job ended
`COMPLETED`, exit `0:0`, unless noted. The HPC paths `/home/msf43/axiom-goal/run_selection.py` and `e_compare.py` are the
originals of the scripts in this folder, which take the same arguments. SATCoP runs used the binary built from SATCoP
`e66dfc8`; Connections was at `9c7c588` for the part-9 SatResetCoP 60 s runs and the SatCoP 60 s baseline and at
`5a4816d` for every later SATCoP run; Prover9 was at `a27a68d` (`part9-test`, `part9-overlap`, `part9-sharbaraz5e20`) and
`98db53d` (`part9-ladder-*`); Vampire's runner was at `0270a14` with a binary from `7770dbe`; E was 3.5.5 (`bdeba77`).

### SATCoP part 9, 60 s runs (`satcop-part9/`)

```
cd /home/msf43/connections && PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-baseline-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections  --policy SatResetCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/baseline-SatResetCoP.csv

cd /home/msf43/connections && PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-sharbaraz5e20-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatResetCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/sharbaraz5-e20-SatResetCoP.csv

cd /home/msf43/connections && PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-random-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatResetCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/random-SatResetCoP.csv

cd /home/msf43/connections && PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-baseline-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections  --policy SatCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/baseline-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-s5e20-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/sharbaraz5-e20-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t9-random-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatCoP --timeout-seconds 60 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9/random-SatCoP.csv
```

### SATCoP part 9, 20 s stage runs for the three-stage schedule (`satcop-part9-t20/`)

A problem counts as proved by a schedule if any of its three 20 s stage runs proves it (`thirds.py`, `seeds.py`). The
plain 60 s SatCoP comparison line is `satcop-part9/baseline-SatCoP.csv` above. Seeds 1 and 2 add `--seed 1` / `--seed 2`
after `--policy ...` and write `*-seed1-*.csv` / `*-seed2-*.csv`.

```
cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20-baseline-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections  --policy SatCoP --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/baseline-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20-s5e20-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatCoP --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20-s5e20-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatResetCoP --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-SatResetCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20-random-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatCoP --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20-random-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatResetCoP --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-SatResetCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s1-baseline-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections  --policy SatCoP --seed 1 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/baseline-seed1-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s1-s5e20-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatCoP --seed 1 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-seed1-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s1-s5e20-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatResetCoP --seed 1 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-seed1-SatResetCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s1-random-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatCoP --seed 1 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-seed1-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s1-random-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatResetCoP --seed 1 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-seed1-SatResetCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s2-baseline-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections  --policy SatCoP --seed 2 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/baseline-seed2-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s2-s5e20-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatCoP --seed 2 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-seed2-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s2-s5e20-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --policy SatResetCoP --seed 2 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/sharbaraz5-e20-seed2-SatResetCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s2-random-SatCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatCoP --seed 2 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-seed2-SatCoP.csv

cd /home/msf43/connections && TMPDIR=/dev/shm PYTHONPATH=/home/msf43/connections/packages:/home/msf43/connections/src SATCOP_LOCATION=/home/msf43/validation-sharbaraz/satcop-e66dfc8 /home/msf43/bin/icelake-batch --intr --cpus 32 --time 00:59:00 --name t20s2-random-SatResetCoP /home/msf43/venv/bin/python /home/msf43/axiom-goal/run_selection.py /home/msf43/axiom-goal/part9-selection.json /home/msf43/connections --model /home/msf43/axiom-goal/models/random.ts --policy SatResetCoP --seed 2 --timeout-seconds 20 --cpu --num-workers 32 --no-wandb --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/satcop-part9-t20/random-seed2-SatResetCoP.csv
```

### Prover9 part 9 (`prover9/part9-*/`)

`part9-test`, `part9-overlap` and `part9-sharbaraz5e20` ran Prover9 `a27a68d`; `part9-ladder-*` ran `98db53d`. The stage
attribution `part9-test/gain-stages.json` was produced afterwards by `stages.py`.

```
cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-test /home/msf43/venv/bin/python -u axiom_selection/compare.py --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --modes Baseline ResumeFilter16Support1 ShuffledResumeFilter16Support1 ResumeSupport1 --model /home/msf43/connections/results/big-train/Sharbaraz1/model/epoch-0100.ts --output /home/msf43/axiom-goal/prover9/part9-test

cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-overlap /home/msf43/venv/bin/python -u axiom_selection/compare.py --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --modes ResumeFilter16Support1 --model /home/msf43/axiom-goal/models/overlap.ts --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --output /home/msf43/axiom-goal/prover9/part9-overlap

cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-s5e20 /home/msf43/venv/bin/python -u axiom_selection/compare.py --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --modes Baseline ResumeFilter16Support1 ShuffledResumeFilter16Support1 --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --output /home/msf43/axiom-goal/prover9/part9-sharbaraz5e20

cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-ladder-s5e20 /home/msf43/venv/bin/python -u axiom_selection/compare.py --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --modes Baseline ResumeLadder ShuffledResumeLadder ResumeLadderSupport1 ShuffledResumeLadderSupport1 --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --output /home/msf43/axiom-goal/prover9/part9-ladder-s5e20

cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-ladder-overlap /home/msf43/venv/bin/python -u axiom_selection/compare.py --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --modes ResumeLadder ResumeLadderSupport1 --model /home/msf43/axiom-goal/models/overlap.ts --output /home/msf43/axiom-goal/prover9/part9-ladder-overlap

cd /home/msf43/TP/Prover9 && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name p9-part9-ladder-sine /home/msf43/venv/bin/python -u axiom_selection/compare.py --tptp /rds/user/msf43/hpc-work/TPTP --seconds 60 --workers 76 --manifest /home/msf43/axiom-goal/manifests/part9-cnf-unsat.json --modes ResumeLadder ResumeLadderSupport1 --model /home/msf43/axiom-goal/models/sine --output /home/msf43/axiom-goal/prover9/part9-ladder-sine
```

### Vampire part 8 (`vampire/part8-*/`; no part-9 run)

```
cd /home/msf43/TP/vampire && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name vampire-part8 /home/msf43/venv/bin/python -u Shell/Torok/experiments/compare.py --manifest /home/msf43/axiom-goal/manifests/vampire-part8.json --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/vampire/part8-sharbaraz5e20 --workers 76 --seconds 60 --algorithm lrs --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --modes baseline resume-filter16 random-resume-filter16 resume-support1 random-resume-support1

cd /home/msf43/TP/vampire && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name vampire-part8-ladder /home/msf43/venv/bin/python -u Shell/Torok/experiments/compare.py --manifest /home/msf43/axiom-goal/manifests/vampire-part8.json --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/vampire/part8-ladder-s5e20 --workers 76 --seconds 60 --algorithm lrs --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --modes baseline sine resume-filter16:8+filter64:8+filter256:8 random-resume-filter16:8+filter64:8+filter256:8

cd /home/msf43/TP/vampire && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name vampire-part8-sinebase /home/msf43/venv/bin/python -u Shell/Torok/experiments/compare.py --manifest /home/msf43/axiom-goal/manifests/vampire-part8.json --tptp /rds/user/msf43/hpc-work/TPTP --output /home/msf43/axiom-goal/vampire/part8-sinebase-s5e20 --workers 76 --seconds 60 --algorithm lrs --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --modes baseline included resume-sine:24 resume-included:24 resume-sine:8+filter16:8+filter64:8
```

### E prover part 8 (`eprover/part8-*/`)

Job `e-part8-s5e20` (40309795) hit its 59 min limit after `Baseline` and `ESInE`; `Ladder` and `ShuffledLadder` did not
finish. `e-part8-overlap` (40312362) was still running when the receipts were collected.

```
cd /home/msf43/axiom-goal && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name e-part8-s5e20 /home/msf43/venv/bin/python -u e_compare.py --tptp /rds/user/msf43/hpc-work/TPTP --workers 76 --seconds 60 --manifest manifests/part8-cnf-unsat.json --modes Baseline ESInE Ladder ShuffledLadder --model /home/msf43/connections/results/big-train/Sharbaraz5/model/epoch-0020.ts --output eprover/part8-s5e20

cd /home/msf43/axiom-goal && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 00:59:00 --name e-part8-overlap /home/msf43/venv/bin/python -u e_compare.py --tptp /rds/user/msf43/hpc-work/TPTP --workers 76 --seconds 60 --manifest manifests/part8-cnf-unsat.json --modes Ladder --model /home/msf43/axiom-goal/models/overlap.ts --output eprover/part8-overlap
```

### Part-8 runs behind the choices

The validation runs follow the same patterns with `part8-selection.json` / `part8-cnf-unsat.json` and outputs under
`satcop-part8-w32/` or `prover9/part8-*/`; receipt operations (all `-intr-20261106`): `p8w32-{baseline,s5e20,random,sine,
overlap,mix0.25,mix0.5,top16,top64,rtop16,rtop64,sharbaraz1-e30,sharbaraz10-e20,sharbaraz15-e20}-SatCoP`,
`p8w32-{sharbaraz5e20,random,sine,overlap,sharbaraz1-e30,sharbaraz10-e20,sharbaraz15-e20,s5e20-seed1,random-seed1}-SatResetCoP`,
`p9-part8-{sharbaraz1,sharbaraz1-variants,filter,random,overlap,torok,s5e20,ladder-s5e20,ladder-overlap,ladder-sine,ladder2-s5e20,ladder2-overlap}`,
`model-quality-part8`, `model-quality-union{,2,3}`, `clean-Sharbaraz5-e{10..100}-0` (the checkpoint sweep).
