# SATResetCoP on StarExec / CASC

This directory packages the **SATResetCoP** connection prover as a
[StarExec](https://github.com/StarExecMiami/StarExec) solver so it can be entered into
[CASC](https://tptp.org/CASC/). Everything needed lives under `CASC/`; the
root project is **not** modified — the prover source is copied into
`SATResetCoP/src/` and patched there.

> **Target = the `containerised` branch of `github.com/StarExecMiami/StarExec`**
> (the Podman-based StarExec deployment for CASC-J13; successor to the now-
> abandoned `StarExec-ARC`; the old `starexec.org` host is decommissioned).
> That branch changes how the *service* runs (jobs execute in Podman
> containers), **not the solver package format**: its `docker/job-entrypoint.sh`
> still runs a standard `bin/starexec_run_*` configuration under `runsolver` and
> passes `STAREXEC_CPU_LIMIT` / `STAREXEC_WALLCLOCK_LIMIT` (which
> `starexec_prover.py` already reads). So **`CASC/SATResetCoP.tar.gz` works
> unchanged** — no per-prover Dockerfile needed. This also matches the CASC-J13
> official rule (delivery as a StarExec `.tgz`).
>
> `CASC/StarExec/` is a checkout of that platform (currently on the
> `containerised` branch), reference only.

---

## TL;DR

```sh
bash CASC/build_package.sh          # builds the portable runtime + tarball
# -> CASC/SATResetCoP.tar.gz
```

Upload `SATResetCoP.tar.gz` to StarExec as a **solver**. It exposes two
configurations, `default` and `SATResetCoP` (identical). StarExec runs

```
bin/starexec_run_default <benchmark.p>
```

and the prover prints an SZS status line, e.g. `% SZS status Theorem for PUZ001+1`.

---

## The two portability problems (and how they're solved)

StarExec's compute nodes are old (historically CentOS 7, glibc 2.17). The
development machine here is glibc 2.43, and SATResetCoP has **two native
dependencies** that would not otherwise run there:

| Dependency | Role | Problem | Solution |
|---|---|---|---|
| `pydical` | CaDiCaL SAT binding | single `.so` built for CPython 3.14 + glibc 2.43 → won't load on old nodes | **Swapped for `python-sat` (PySAT)**, which ships `manylinux2014` wheels. New `SATSolver.PySAT` backend in `src/connections/calculi/classicalsat.py`. |
| `swipl` (SWI-Prolog) | FOF → CNF clausification (leanCoP `translation/`) | not installed on nodes | **Bundled from conda-forge** (built for glibc 2.17) inside the runtime. |

Both are delivered by a single relocatable **conda-forge environment** built
with a static `micromamba` (no root, no pre-installed conda). Because we no
longer depend on the CPython-3.14-only `pydical.so`, the bundle uses a more
mature **Python 3.13**.

### Is the SAT swap sound?

Yes. The connection search only asks the SAT engine whether the grounded
"shadow" clause set is UNSAT/SAT. Any complete SAT solver returns the same
verdict, so proofs are unchanged. Only the concrete satisfying *assignments*
(hence model-guided tie-breaking) can differ, which may change step counts /
which problems close within the time limit — but never correctness. On the
smoke tests both backends returned identical results.

---

## Files

```
CASC/
├── build_package.sh          # assembles the runtime + tarball (run this)
├── README.md
└── SATResetCoP/              # <- packaged solver (tar root)
    ├── bin/
    │   ├── starexec_run_default      # StarExec entry point
    │   └── starexec_run_SATResetCoP  # named config (execs the default)
    ├── starexec_prover.py    # SZS/CASC front-end (time limit + status mapping)
    ├── src/                  # copied prover source, with the PySAT backend
    │   ├── pycop.py  cli.py  connections/  strategies/  translation/
    ├── tptp-root/            # placeholder TPTP library root (see below)
    └── python/               # bundled runtime  [created by build_package.sh]
```

---

## `starexec_prover.py`

The CASC front-end that differs from `pycop.py` in three ways that matter:

1. **SZS output** instead of a Python `dict`:
   * proved a problem **with a conjecture** → `% SZS status Theorem`
   * proved a **CNF / no-conjecture** problem → `% SZS status Unsatisfiable`
   * ran out of time → `% SZS status Timeout`
   * anything else (parse error, no proof found, recursion limit) → `% SZS status GaveUp`

   Theorem-vs-Unsatisfiable is decided **structurally** (does the problem
   contain a `conjecture`-role formula), never from the `% Status` header.
   `--proof` additionally prints the refutation witness inside an
   `% SZS output start/end CNFRefutation` block.

2. **Self-imposed deadline.** It reads the limit from `--cpu-limit`, else
   `$STAREXEC_WALLCLOCK_LIMIT` / `$STAREXEC_CPU_LIMIT` (default 300 s), keeps a
   small safety margin, and a `SIGALRM` watchdog prints `Timeout` gracefully
   just before `runsolver` kills the process.

3. **PySAT backend by default** (`--sat-solver PySAT`).

Run it locally:

```sh
python CASC/SATResetCoP/starexec_prover.py --cpu-limit 60 some-problem+1.p
python CASC/SATResetCoP/starexec_prover.py --proof PUZ001-1.p
```

---

## TPTP axioms / `include()` directives

Most CASC problems `include('Axioms/...')`. Those resolve relative to the
`$TPTP` root. The run script honours an existing `$TPTP`; otherwise it falls
back to the bundled `tptp-root/` (empty — enough for self-contained problems,
whose clausifier still *requires* the variable to be set).

**If the competition does not set `$TPTP`**, drop the TPTP `Axioms/` tree into
`SATResetCoP/tptp-root/Axioms/` before building, or point the run script's
fallback at wherever StarExec stages the axioms.

---

## Running / submitting

There is no public hosted StarExec to log into. Use one of:

**A. Submit to CASC.** Build the tarball and give it to the CASC organisers,
who run it on their StarExec (built from `github.com/StarExecMiami/StarExec`).

**B. Deploy StarExec yourself** from that repo (see its
`web-docs/installation.html`), then on your instance:

1. Build: `bash CASC/build_package.sh` → `CASC/SATResetCoP.tar.gz`.
2. *Solvers → Upload* the tarball. Do **not** tick "run build script"
   (the runtime is pre-built; there is no `starexec_build`).
3. Two configurations appear: `default` and `SATResetCoP`.
4. Create a job over a TPTP benchmark space, pick a CPU/wall limit; StarExec
   enforces it with `runsolver` and scores the `% SZS status` line.

**C. Pre-flight without StarExec** — the run script *is* the StarExec contract,
so you can exercise it directly (this is what was validated here):

```sh
tar xzf CASC/SATResetCoP.tar.gz -C /tmp/se
/tmp/se/bin/starexec_run_default /path/to/PROBLEM.p 300   # -> % SZS status ...
```

---

## What has been verified here

Prover logic, with the real prover + `python-sat` + `swipl`:

| Case | Result |
|---|---|
| CNF refutation | `% SZS status Unsatisfiable` |
| FOF w/ conjecture (clausified via swipl) | `% SZS status Theorem` (+ proof block) |
| unparseable input | `% SZS status GaveUp` |
| satisfiable set + short limit | `% SZS status Timeout` (watchdog fires before kill) |
| **pydical absent, python-sat present** (bundle scenario) | `% SZS status Unsatisfiable` |
| PySAT vs pydical on same problem | identical verdict & step count |

Built bundle (`build_package.sh`), running through the **bundled** runtime via
`bin/starexec_run_default`:

| Check | Result |
|---|---|
| CNF via bundled Python + python-sat | `% SZS status Unsatisfiable` |
| FOF via bundled `swipl` clausifier (`SWI_HOME_DIR`) | `% SZS status Theorem` |
| Bundle self-test in `build_package.sh` | passes |
| Real library problems (PUZ001+1, PUZ005+1, PUZ031+1) | `% SZS status Theorem` |
| `include()`-based problem (ALG213+1) — axioms resolved & clausified | ran to `Timeout` (no error) |
| **Relocation**: untar to a fresh path, run with a clean `env -i` (CNF + FOF) | `Unsatisfiable` / `Theorem` |

### Portability floor (measured on the built bundle)

* **glibc ≥ 2.17** — the max `GLIBC_` symbol required across Python 3.13,
  `libswipl`, and python-sat's `pysolvers`/`pycard` is `GLIBC_2.17` (Python;
  swipl and python-sat need only 2.14). That means **CentOS 7 and newer**.
* **libstdc++/libgcc are bundled** (`python/lib/libstdc++.so.6.0.34`,
  `GLIBCXX_3.4.34`; python-sat needs `GLIBCXX_3.4.21`) and exposed via
  `LD_LIBRARY_PATH` in the run script — so it does **not** depend on the
  node's (ancient) system libstdc++.
* glibc itself is **not** bundled; it uses the node's, which is why 2.17 is the
  floor.

Tarball size: ~127 MB.

## Equivalence benchmark: original (pydical) vs packaged (PySAT)

To confirm the SAT-backend swap changes nothing that matters, 23 varied TPTP
problems were run through both backends **with the same interpreter** (only
`--sat-solver` differs): original = root `pycop.py` (pydical), packaged =
`CASC/SATResetCoP/src/pycop.py --sat-solver PySAT`.

| Group | Problems | Outcome |
|---|---|---|
| FOF theorems (`+1`, swipl-clausified) | PUZ001+1, PUZ005+1, SYN000/001/036/040/041/044/045/046+1 (10) | verdict identical 10/10 |
| CNF & other (`-1`, pure SAT) | PUZ001/002/003/004/005/006-1, SYN000/006-1 (8) | verdict identical 8/8 |
| Hard / unsolvable-in-limit | PUZ001+2, ALG212/213/215/217+1 (5) | both `Timeout`, no crash, 5/5 |

Findings:

* **Verdict: 23/23 identical.** The swap never changed an answer, never
  crashed, never produced a bogus status. (This is what CASC scores.)
* **Steps: identical in 13/18 solved.** The 5 that differ move by small amounts
  in *both* directions (PySAT sometimes fewer, sometimes more) — the benign,
  expected effect of a different satisfying assignment breaking search ties
  differently. No runaway search.
* **Speed: comparable.** CNF (which isolates the SAT engine) is near-identical,
  e.g. PUZ005-1 `572 ms` (pydical) vs `496 ms` (PySAT). FOF time is dominated
  by the *identical* swipl clausification, so totals track step counts and stay
  within ~1.5× either way. No order-of-magnitude regressions.

Reproduce with the `compare*.sh` scripts used for this (per-problem `timeout`,
wall-time in ms, step counts parsed from the prover's output).

## Containerised job runner (Alpine/musl)

The `containerised` branch runs each job in an **Alpine 3.21 (musl libc)**
container (`docker/job-runner.Dockerfile`), where glibc binaries do **not** load
under `gcompat`, and Alpine has no `swi-prolog` package. The bundle handles this
by shipping a small **glibc runtime** (loader + core libs) at `python/glibc/`
and invoking Python/SWI-Prolog through it via `python/shim/{python3,swipl}`
(the loader is called with an explicit `--library-path`). The result is
self-contained on **any** host libc — verified in both Alpine (podman) and on a
glibc host. Two related details:

* the shim runs Python with `-s -E` so it never imports host packages
  (`~/.local`, `$PYTHONPATH`);
* PodmanBackend mounts the benchmark as the fixed name
  `/starexec/input/benchmark` (no `.p`/`+`/`-`), so `starexec_prover.py`
  content-sniffs FOF vs CNF and normalises the filename before parsing.

Test it yourself (needs a running docker/podman daemon):

```sh
bash CASC/test_in_container.sh [problem.p] [limit]
# runs the bundle inside alpine:3.21 exactly as job-entrypoint.sh does
# -> expect "% SZS status Theorem for benchmark"
```

## Caveats

* Relocation + clean-environment execution is **verified locally** (untar to a
  fresh path, run under `env -i`). The only thing not reproducible here is the
  node's exact glibc — but the measured floor is 2.17, so CentOS 7+ is safe. As
  a belt-and-braces step you can still run one FOF job on a StarExec test space
  before the competition; if `swipl` ever fails to find its home, add a
  `starexec_build` that runs `python/bin/conda-unpack`.
* **Proof output is a connection/SAT witness**, not a formal TPTP derivation —
  adequate for assurance-level divisions; a verifiable proof is future work.
* **`Non-Theorem` is reported as `GaveUp`**, not `(Counter)Satisfiable`, to
  stay sound (the search is a refutation procedure). Change the mapping in
  `starexec_prover.py:_map_status` if a completeness argument justifies it.
