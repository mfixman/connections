from enum import StrEnum

from axiom_prediction.prover_cores.provers import eprover, iprover, prover9, vampire

class Prover(StrEnum):
    Prover9 = "Prover9"
    E = "E"
    Vampire = "Vampire"
    IProver = "iProver"

modules = {Prover.Prover9: prover9, Prover.E: eprover, Prover.Vampire: vampire, Prover.IProver: iprover}
