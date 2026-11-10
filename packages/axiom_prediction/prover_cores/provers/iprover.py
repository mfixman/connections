from axiom_prediction.prover_cores.locations import external
from axiom_prediction.prover_cores.provers.tptp_input import proved, used_indices, write_input

label_semantics = "input clauses used in an iProver proof (default settings)"

def command(path, seconds):
    return [str(external / "iprover/iproveropt"), "--time_out_real", str(int(seconds)), str(path)]
