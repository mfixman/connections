# E with correctly printed filtered problems (part 8, CNF), plus ranking transfer of fixed scorers to each prover's cores.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
M=/rds/user/msf43/hpc-work/frontier-gaps-20261109/manifests
B=/home/msf43/connections/results/big-train
PY=/home/msf43/venv/bin/python
S5=$B/Sharbaraz5/model/epoch-0020.ts
OVERLAP=/rds/user/msf43/hpc-work/axiom-goal/models/overlap.ts
SKIP=/rds/user/msf43/hpc-work/epoch-stages-20261111/ost114-problems.json
export TMPDIR=/dev/shm MKL_NUM_THREADS=1
cpus=($($PY -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
n=${#cpus[@]}
rank_cpus=$(IFS=,; echo "${cpus[*]:$((n - 30))}")
stage_cpus=$(IFS=,; echo "${cpus[*]:0:$((n - 30))}")
mkdir -p $R/ranking $R/E/controls $R/E/Sharbaraz5-e20 $R/E/overlap
echo "cpus $n: stages on $((n - 30)), ranking on 30; $(date -u +%H:%M:%S)"
(
    export OMP_NUM_THREADS=6
    scorers="$S5 $B/Sharbaraz15/model/epoch-0020.ts $OVERLAP $B/Subutai-Prover9-CNF/model/epoch-0050.ts"
    for p in Prover9 E Vampire iProver; do
        taskset -c $rank_cpus $PY -u $R/rank_transfer.py /rds/user/msf43/hpc-work/subutai/$p/p8/rows.jsonl $R/ranking/$p.csv 6 $scorers > $R/ranking/$p.log 2>&1 &
    done
    taskset -c $rank_cpus $PY -u $R/rank_transfer.py $B/datasets/Sharbaraz5 $R/ranking/SATCoP.csv 6 $scorers > $R/ranking/SATCoP.log 2>&1 &
    wait
    echo "=== ranking done $(date -u +%H:%M:%S)"
) &
export OMP_NUM_THREADS=1
stage() {
    echo "=== $* $(date -u +%H:%M:%S)"
    taskset -c $stage_cpus $PY -u $R/stages2.py --prover eprover --manifest $M/part8.json --skip $SKIP --workers $((n - 30)) "$@" || echo "FAILED: $*"
}
stage --kind filters --model $S5 --output $R/E/Sharbaraz5-e20/eprover-filters.csv
stage --kind plain --output $R/E/controls/eprover-plain.csv
stage --kind sine --output $R/E/controls/eprover-sine.csv
stage --kind filters --shuffled --output $R/E/controls/eprover-filters-shuffled.csv
stage --kind filters --model $OVERLAP --output $R/E/overlap/eprover-filters.csv
wait
echo "=== done $(date -u +%H:%M:%S)"
