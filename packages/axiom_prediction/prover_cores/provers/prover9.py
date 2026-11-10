# Clause i carries label(weighted_input_{i}), which Prover9 keeps on the input lines of its proof.
import re

from axiom_prediction.prover_cores.clauses import clause_text, prover9
from axiom_prediction.prover_cores.locations import external

label_semantics = "input clauses used in a Prover9 proof (default settings)"

def write_input(matrix, conjectures, flip, path, seconds):
    limits = [f"assign(max_seconds,{int(seconds)}).", "assign(max_megs,512)."]
    quiet = ["clear(print_given).", "clear(print_initial_clauses).", "clear(print_kept)."]
    clauses = [
        f"{clause_text(clause, prover9, flip)} # label(weighted_input_{index})."
        for index, clause in enumerate(matrix.clauses)
    ]

    lines = ["set(prolog_style_variables).", *limits, *quiet, "formulas(assumptions).", *clauses, "end_of_list.", ""]
    with open(path, "w") as stream:
        stream.write("\n".join(lines))

def command(path, seconds):
    return [str(external / "Prover9/provers.src/prover9"), "-f", str(path)]

def proved(output):
    return "THEOREM PROVED" in output

def used_indices(output):
    start, end = output.find("= PROOF ="), output.find("= end of proof =")
    return sorted(set(map(int, re.findall(r"weighted_input_(\d+)", output[start:end]))))
