#!/bin/sh
# Regenerate the recorded SATResetCoP example results and TSTP certificates.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
: "${TPTP:?Set TPTP to the TPTP library root}"

for problem in SYN001+1 SYN036+1 SYN040+1; do
    input="$TPTP/Problems/SYN/$problem.p"
    python "$ROOT/pycop.py" \
        --strategy SATResetCoP \
        --tptp "$TPTP" \
        --max-steps 100000 \
        "$input"
done > "$ROOT/CASC/certificate-examples/pycop.txt"

for problem in SYN001+1 SYN036+1 SYN040+1; do
    input="$TPTP/Problems/SYN/$problem.p"
    python "$ROOT/CASC/SATResetCoP/starexec_prover.py" \
        --strategy SATResetCoP \
        --sat-solver PySAT \
        --proof \
        --cpu-limit 30 \
        "$input" > "$ROOT/CASC/certificate-examples/$problem.tstp"
done
