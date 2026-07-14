# SATResetCoP certificate examples

These are reproducible outputs of the root `pycop.py` and the packaged CASC
front end. They are deliberately small TPTP `SYN` theorem problems:

| Problem | Root `pycop.py` result | CASC result | GDV 2.0 |
|---|---:|---|---|
| `SYN001+1` | `Theorem (SAT)`, 0 steps | `Theorem` + certificate | `VerifiedGood` (4.95 CPU s) |
| `SYN036+1` | `Theorem (SAT)`, 1824 steps | `Theorem` + certificate | `VerifiedGood` (61.04 CPU s) |
| `SYN040+1` | `Theorem (SAT)`, 0 steps | `Theorem` + certificate | `VerifiedGood` (6.10 CPU s) |

`pycop.txt` records the root prover's ordinary output. Each `.tstp` file is
the complete `--proof` output of the packaged CASC front end, including its
SZS delimiters.

The root prover uses its normal CaDiCaL configuration; the CASC package uses
PySAT, because that is the portable competition configuration. Both runs use
the `SATResetCoP` strategy and the same `$TPTP` problem files.

GDV was given each derivation through SystemOnTSTP. It verified all derived
formulae, including the `status(esa)` clausification and final SAT
refutation. As the web form was given the derivation rather than a separately
uploaded benchmark, GDV warns that its verification of the input leaves is
incomplete.

Regenerate the files with:

```sh
TPTP=/path/to/TPTP CASC/certificate-examples/regenerate.sh
```
