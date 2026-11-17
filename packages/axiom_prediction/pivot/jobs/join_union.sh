# Joins the Sharbaraz5 dataset (SATCoP and SATResetCoP cores, 5 seeds) with the four Subutai datasets (each prover's
# own proofs) into datasets/Sharbaraz5+Subutai: every proof stays its own example, graphs are deduplicated by join.
set -u
C=/home/msf43/connections
D=$C/results/big-train/datasets
cd $C
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u axiom_predictor.py join $D/Sharbaraz5 $D/Subutai-Prover9 $D/Subutai-E $D/Subutai-Vampire $D/Subutai-iProver \
    --dataset "$D/Sharbaraz5+Subutai" --num-workers 32
echo "join exit $?"
ls -la "$D/Sharbaraz5+Subutai"
/home/msf43/venv/bin/python -c "
import json; m = json.load(open('$D/Sharbaraz5+Subutai/metadata.json'))
print({k: m.get(k) for k in ('problems', 'graphs', 'examples', 'failures', 'provers', 'seeds')})"
echo "done $(date -u +%H:%M:%S)"
