# External provers

Submodules point at the revisions used for the axiom-selection experiments (handoff-2026-11-09/).

| Directory | Source | Revision | Notes |
|---|---|---|---|
| Prover9 | github.com/mfixman/Prover9, branch axiom_selection | 4a70421 | axiom_selection/ runner, ladders, SInE and dual-ladder modes; built with axiom_selection/build.py against libtorch |
| satcop | github.com/mfixman/satcop, branch axiom_selection | 177002a | --weights, --weight-mix (0ecdb06), --filter-top (177002a); main results used the e66dfc8 binary |
| vampire | github.com/mfixman/vampire, branch axiom_selection | 0270a14 | Torok integration in Shell/Torok; binary built from 7770dbeed |
| eprover | github.com/eprover/eprover, master | bdeba77 (E 3.5.5) | unmodified; selection is done outside the prover by experiments/axiom_goal/e_compare.py |
| iprover | gitlab.com/korovin/iprover, master | 6043163 (3.9.4) | unmodified, built without Z3 (build-provers.sh) |
| SPASS | spass-prover.org 3.9 tarball | 3.9 | unmodified source, vendored (no git upstream); build outputs ignored |

E, SPASS and iProver are driven as black boxes: the runner writes a filtered CNF file with the top-k ranked
axioms, runs the unmodified binary on it, and pauses/resumes the unguided baseline with SIGSTOP/SIGCONT.
On the HPC these directories were built under ~/TP, which now holds symlinks into this directory.
