from axiom_prediction.prover_cores.locations import external
from axiom_prediction.prover_cores.provers.tptp_input import proved, used_indices, write_input

label_semantics = "input clauses used in a Vampire proof (LRS, no SInE)"

def command(path, seconds):
    binary = external / "vampire/build/vampire"
    limits = ["--time_limit", f"{int(seconds)}s", "--memory_limit", "2048"]
    search = ["--mode", "vampire", "--saturation_algorithm", "lrs", "--sine_selection", "off"]
    return [str(binary), *limits, *search, "--proof", "tptp", str(path)]
