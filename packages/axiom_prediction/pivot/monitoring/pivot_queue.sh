#!/bin/bash
# Runs the given intr CPU jobs one after another: gated submission, then waits for the job's "done" line (file reads
# over ssh, no Slurm commands). usage: pivot_queue.sh "KEY|NAME|SCRIPT ARGS" ...
cd /home/debian/epoch-sweep
for spec in "$@"; do
    IFS='|' read -r key name script <<< "$spec"
    log=pivot/submit_$key.log
    python3 submit_gated.py "pivot-$key=cd /home/msf43/connections && /home/msf43/bin/icelake-batch --intr --cpus 76 --time 0:59:00 --name $name bash /rds/user/msf43/hpc-work/pivot-20261114/jobs/$script" > $log 2>&1
    out=$(grep -o "/home/msf43/tmp/$name-[0-9]*/stdout.out" $log | head -1)
    echo "$(date -u +%FT%TZ) $key submitted: $(head -c 120 $log)" >> pivot/queue.log
    if [ -z "$out" ]; then
        echo "$(date -u +%FT%TZ) $key: no job directory in the submission reply; queue stopped" >> pivot/queue.log
        exit 1
    fi

    until ssh -F ~/.ssh/config -o BatchMode=yes -o ConnectTimeout=120 hpc "grep -q '^done' $out || grep -qi 'time limit\|cancelled' ${out%stdout.out}stderr.err" 2>/dev/null; do
        sleep 120
    done

    echo "$(date -u +%FT%TZ) $key ended: $(ssh -F ~/.ssh/config -o BatchMode=yes -o ConnectTimeout=120 hpc "grep -v '^{' $out | tail -1; grep -i 'time limit\|cancelled' ${out%stdout.out}stderr.err | head -1")" >> pivot/queue.log
done

echo "$(date -u +%FT%TZ) queue finished" >> pivot/queue.log
