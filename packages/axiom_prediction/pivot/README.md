# Pivot investigation scripts (14-17 November 2026)

Research scripts behind the oracle, ladder, SInE and no-equality-axiom results. Results, notes and the handover are in
handoff-2026-11-16/ (local) and /rds/user/msf43/hpc-work/pivot-20261114/ (HPC). They run from that RDS folder and
contain absolute HPC paths; they are kept here so that the code is versioned, not as a package.

- library_stages.py: stage runner (filtered top-k, external SInE settings, plain, fallback modes) for seven provers.
- real_schedule.py: executes a schedule end to end within 60 s, to check the rebuilt ones.
- oracle_cores.py, plain_fallback.py, eclausify.py, noeq_lists.py: oracle transfer, fallback plain runs, E clausification, problem lists.
- *_table.py, *_pool.py, pareto*.py, noeq_*.py, fair_table.py: tables rebuilt from stage rows.
- jobs/: Slurm job scripts. epoch_sweep/: the epoch-sweep runner and simulator. monitoring/: the VPS Slurm gate, poller and submitters.
