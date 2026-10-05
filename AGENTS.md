# Connections Prover

* Read ~/AGENTS.md to start.
* This is a repo modified by multiple people. Focus on areas modified by myself.
* Keep changes within `packages/axiom-predictor/`. Only make small changes elsewhere when necessary for the axiom predictor; do not perform unrelated cleanup outside it.
* I prefer to have simple Python invocation:
    * Requirements should be in requirements.txt and be installable by pip.
    * The programs should be launchable by `python some/directory/file.py`, rather than using uv or the like.
* I will likely ask you to work on the axiom selection.
* Prefer clear, direct code over exhaustive edge-case machinery. Keep checks that protect proof correctness and saved results.

## Axiom predictor style

* Use lowercase snake_case for constants; do not define constants in ALL_CAPS.
* Declare constants in the narrowest scope that uses them; keep only shared constants at module scope.
* Use unit symbols in names, such as `default_timeout_s` and `poll_interval_ms`, rather than spelling out seconds or milliseconds. Preserve established external field names when needed for compatibility.
* Do not use `@property`; use ordinary methods or attributes.
* Avoid `object` for initialization or attribute assignment; use ordinary `self` assignments and simple classes.
* Do not define `__getstate__`, `__setstate__`, or `__post_init__`; use explicit constructors and named helpers instead.
* Do not use `__all__` unless an actual import/export requirement needs it.
* Prefer direct, simple code and short comments. Avoid unnecessary wrappers and boilerplate.
* Keep only essential block comments and docstrings, usually one short sentence explaining non-obvious behavior. Remove commentary that repeats the code.
* Use `#` comments instead of single-line docstrings.
* Prefer `map` and `filter` for simple transformations using existing functions; materialize a list only when needed.
* Avoid `TYPE_CHECKING` blocks; keep runtime imports local when lazy loading matters.
* Do not prefix ordinary names with `_`. Preserve names required by Python or inherited interfaces.
* Avoid routine hyperparameter validation. Keep checks for non-obvious combinations such as Base mode with top_k, and for data or checkpoint mismatches.
* Trust internally produced clause IDs and data types; avoid separate validators for routine range, uniqueness, and type checks.

## Repository checkouts and synchronization

* Unless necessary, keep one clone and one working checkout of each repository per devserver, including this local computer. Use the main checkout for new work and job submissions; do not create extra clones or worktrees merely to pin each commit.
* Keep all branches with remotes synchronized on every devserver. Fetch their upstreams, fast-forward branches that are behind, and push intended local commits. Verify the actual local branches against their upstreams, not just the remote-tracking refs. Preserve other people's work; resolve divergence without discarding commits or force-pushing.
* Commit and push intended changes made on this computer, then pull them into the main HPC checkout. Keep the main branch synchronized while experiments use pinned code. Do not automatically commit other collaborators' unreviewed edits.
* If the user plans to modify a prover while long experiments are pending, pin those experiments to the recorded commit. Share one temporary checkout per required revision across jobs where possible; future pulls update the main checkout, never an experiment's pinned checkout or timeout continuation. Keep each experiment's dataset, cache, model outputs, and W&B identity separate.
* An extra checkout is justified only by a concrete need, such as protecting code still required by submitted jobs. Record its reason, revision, dependent job IDs, and cleanup condition before creating it. Prefer a designated temporary directory rather than additional directories in the home directory.
* Before removing an extra checkout, check the scheduler and submission/resume manifests for dependencies and preserve unique edits, results, and credentials. Keep credentials private and outside Git; ensure the main checkout has the W&B secrets file before deleting another copy.
* Remove unnecessary extra checkouts promptly. If a submitted job or its authorized timeout continuation still needs one, record deferred cleanup and delete it once that dependency ends. Never change or delete a checkout underneath a submitted job.

## HPC experiments

* Submit GPU jobs through `~/bin/ampere-batch` and CPU jobs through `~/bin/icelake-batch` on the HPC. CPU model jobs need `--cpu`; `--multiprocess` enables parallel work and adaptive inference batching.
* Preserve the authenticated SSH master. Use `ssh -F ~/.ssh/config -o BatchMode=yes hpc`; never close the master casually, since another login may require a token from the user.
* Keep GPU and CPU comparisons running in parallel unless the user explicitly requests cancellation. SmallFull runs take priority over evaluation.
* Commit and push code fixes locally, then pull on the HPC. Use the main checkout for subsequent work whenever safe; preserve any checkout still required by submitted jobs and their authorized timeout continuations. Create a temporary checkout only when this protection makes it necessary, following the cleanup rules above.
* Use `--output` for resumable runs/evaluations, with `--csv` for CSV. Preserve the output and its `.resume` directory together. CSV/JSONL mismatches must fail loudly.
* Resume only confirmed scheduler timeouts when authorized, using the same command, output, checkout, and allocation limit. Carry the policy to replacement IDs. Never restart completed or user-cancelled jobs, or restart solely because a monitoring request timed out.
* Record submission intent before submitting. If submission outcome is uncertain, reconcile the manifest and scheduler before retrying to avoid duplicate jobs.
* Inspect both scheduler state and current logs/results. A populated result file does not prove a job completed successfully; verify its terminal state, exit code, unique problem coverage, and input errors.
* Resume skips previously recorded input errors too. Retry failed inputs into separate output files with fixed code, then produce an audited consolidated file; preserve the original results.
* NoModel must instantiate neither a neural predictor nor GNN embeddings. Its proof search can still use multiprocessing.
* For these experiments, send immediate Pushbullet notifications for failures and corrective actions, and status reports every hour. Read the token from `~/secrets/pushbullet_token`; never print it.

## Monitoring handoff (2026-10-01)

* The user stopped monitoring; the goal is paused and the local watcher was stopped. Do not resume monitoring or automatic submissions without a new instruction. HPC jobs were left intact.
* Saved watcher state and code are in `/tmp/hpc-six-monitor/` on this VPS. Inspect `state.json` for current job IDs, replacements, commands, and timeout policies; verify them against the HPC before acting. Stop the watcher before editing its state, since it holds state in memory.
* HPC results and submission manifests are under `/home/msf43/connections/results/resumable-20261001/`. Completed CPU comparison files are `SmallFull-run-multiprocess-cpu-complete.csv`, `DefaultFull-run-multiprocess-cpu-complete.csv`, and `NoModel-run-cpu-complete.csv`; each covers 3,573 inputs including repairs.
* GPU comparison jobs use pinned checkout `/home/msf43/connections-hpc-6f06365`. Parser recursion and CNF truth-constant fixes are available in `66ff4ab` and were verified by CPU repairs. GPU inputs failing on the older checkout still need separate repairs before claiming the comparisons are complete.
* Two CPU evaluations were submitted: SmallFull `37078214` and DefaultFull `37078215`, using fixed checkout `/home/msf43/connections-hpc-66ff4ab`, `--cpu --multiprocess`, split 10 parts 8/9, 59-minute limits, and separate resumable CSV outputs. Treat these IDs as historical pointers and check current scheduler state.
