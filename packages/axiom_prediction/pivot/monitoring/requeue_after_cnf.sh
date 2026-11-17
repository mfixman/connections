#!/bin/bash
# Takes over from the running queue once it has submitted SInE-p8-cnf: waits for that job, then runs the part-9 SInE
# control jobs, the baseline check and the dataset join, in that order.
cd /home/debian/epoch-sweep
until grep -q "SInE-p8-cnf submitted" pivot/queue.log; do sleep 30; done
kill "$1"
out=$(grep -o "/home/msf43/tmp/Sharbaraz5-e20-SInE-vs-ladder-CNF-p8-[0-9]*/stdout.out" pivot/submit_SInE-p8-cnf.log | head -1)
if [ -z "$out" ]; then
    echo "$(date -u +%FT%TZ) requeue: no output path for SInE-p8-cnf; stopped" >> pivot/queue.log
    exit 1
fi

until ssh -F ~/.ssh/config -o BatchMode=yes -o ConnectTimeout=120 hpc "grep -q '^done' $out || grep -qi 'time limit\|cancelled' ${out%stdout.out}stderr.err" 2>/dev/null; do
    sleep 120
done

echo "$(date -u +%FT%TZ) SInE-p8-cnf ended: $(ssh -F ~/.ssh/config -o BatchMode=yes -o ConnectTimeout=120 hpc "tail -1 $out")" >> pivot/queue.log
pivot/pivot_queue.sh "SInE-p9-fof1|Sharbaraz5-e20-SInE-vs-ladder-FOF-p9-Prover9-SPASS|sine_p9.sh fof1" \
    "SInE-p9-fof2|Sharbaraz5-e20-SInE-vs-ladder-FOF-p9-E-iProver-Vampire|sine_p9.sh fof2" \
    "SInE-p9-cnf|Sharbaraz5-e20-SInE-vs-ladder-CNF-p9|sine_p9.sh cnf" \
    "Baselines-FOF-p8-p9|Baselines-FOF-p8-p9|baselines_fof.sh" \
    "Join-Sharbaraz5+Subutai|Join-Sharbaraz5+Subutai|join_union.sh"
