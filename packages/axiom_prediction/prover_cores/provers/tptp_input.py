# Shared by the provers that read TPTP: clause i is named c{i}, and proofs cite it as file(..., c{i}).
import re

from axiom_prediction.prover_cores.clauses import clause_text, tptp

def write_input(matrix, conjectures, flip, path, seconds):
    with open(path, "w") as stream:
        for index, clause in enumerate(matrix.clauses):
            role = "negated_conjecture" if index in conjectures else "axiom"
            stream.write(f"cnf(c{index}, {role}, ({clause_text(clause, tptp, flip)})).\n")

def proved(output):
    return bool(re.search(r"SZS status (Unsatisfiable|Theorem|ContradictoryAxioms)", output))

def used_indices(output):
    return sorted(set(map(int, re.findall(r"file\('[^']*',\s*'?c(\d+)'?\)", output))))
