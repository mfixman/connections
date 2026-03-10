#!/usr/bin/env bash
# compare_strategies.sh — compare two strategies across a set of problems.
#
# Usage:
#   ./compare_strategies.sh [DIR] [MAX_STEPS] [TIMEOUT]
#
# Examples:
#   ./compare_strategies.sh
#   ./compare_strategies.sh tests/cnf_problems 100000 60
#   ./compare_strategies.sh comparisons/classical 50000 30

DIR=${1:-tests/cnf_problems}
MAX_STEPS=${2:-50000}
TIMEOUT=${3:-30}
STRAT1=${4:-FirstAction}
STRAT2=${5:-SmartCut}

PASS='\033[32m'  # green
FAIL='\033[31m'  # red
WIN='\033[1;33m' # bold yellow
RESET='\033[0m'

solved1=0; solved2=0; wins1=0; wins2=0; ties=0; total=0

printf "%-30s  %-8s  %-8s  %-6s\n" "Problem" "$STRAT1" "$STRAT2" "Delta"
printf '%s\n' "$(printf '%.0s-' {1..65})"

for f in "$DIR"/*.p; do
    name=$(basename "$f")
    total=$((total + 1))

    out1=$(timeout "$TIMEOUT" python pycop.py --logic ClassicalCut \
        --strategy "$STRAT1" --max-steps "$MAX_STEPS" --print-ratio 0 \
        "$f" 2>/dev/null)
    out2=$(timeout "$TIMEOUT" python pycop.py --logic ClassicalCut \
        --strategy "$STRAT2" --max-steps "$MAX_STEPS" --print-ratio 0 \
        "$f" 2>/dev/null)

    steps1=$(python3 -c "import re,sys; m=re.search(r\"'steps': (\\d+)\", sys.stdin.read()); print(m.group(1) if m else '')" <<< "$out1")
    steps2=$(python3 -c "import re,sys; m=re.search(r\"'steps': (\\d+)\", sys.stdin.read()); print(m.group(1) if m else '')" <<< "$out2")
    thm1=$(echo "$out1" | grep --color=never -c "Theorem")
    thm2=$(echo "$out2" | grep --color=never -c "Theorem")

    [ "$thm1" -eq 1 ] && solved1=$((solved1 + 1))
    [ "$thm2" -eq 1 ] && solved2=$((solved2 + 1))

    # Format columns
    if [ "$thm1" -eq 1 ]; then col1="${PASS}${steps1}${RESET}"; else col1="${FAIL}timeout${RESET}"; fi
    if [ "$thm2" -eq 1 ]; then col2="${PASS}${steps2}${RESET}"; else col2="${FAIL}timeout${RESET}"; fi

    delta="both timeout"
    if [ "$thm1" -eq 1 ] && [ "$thm2" -eq 1 ] && [ -n "$steps1" ] && [ -n "$steps2" ]; then
        if [ "$steps2" -lt "$steps1" ]; then
            pct=$(( (steps1 - steps2) * 100 / steps1 ))
            delta="${WIN}-${pct}%${RESET}"
            wins2=$((wins2 + 1))
        elif [ "$steps1" -lt "$steps2" ]; then
            pct=$(( (steps2 - steps1) * 100 / steps2 ))
            delta="${FAIL}+${pct}%${RESET}"
            wins1=$((wins1 + 1))
        else
            delta="tie"
            ties=$((ties + 1))
        fi
    elif [ "$thm2" -eq 1 ] && [ "$thm1" -eq 0 ]; then
        delta="${WIN}${STRAT2} only${RESET}"
        wins2=$((wins2 + 1))
    elif [ "$thm1" -eq 1 ] && [ "$thm2" -eq 0 ]; then
        delta="${FAIL}${STRAT1} only${RESET}"
        wins1=$((wins1 + 1))
    fi

    printf "%-30s  %-20b  %-20b  %b\n" "$name" "$col1" "$col2" "$delta"
done

printf '%s\n' "$(printf '%.0s-' {1..65})"
printf "Solved: %s=%d/%d  %s=%d/%d  |  Wins: %s=%d  %s=%d  Ties=%d\n" \
    "$STRAT1" "$solved1" "$total" \
    "$STRAT2" "$solved2" "$total" \
    "$STRAT1" "$wins1" \
    "$STRAT2" "$wins2" \
    "$ties"
