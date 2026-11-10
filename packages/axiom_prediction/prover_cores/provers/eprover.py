from axiom_prediction.prover_cores.locations import external
from axiom_prediction.prover_cores.provers.tptp_input import proved, used_indices, write_input

label_semantics = "input clauses used in an E proof (--auto, default settings)"

def command(path, seconds):
    binary = external / "eprover/bin/eprover"
    return [str(binary), "--auto", "--tstp-format", "-s", "--proof-object", f"--cpu-limit={int(seconds)}", str(path)]
