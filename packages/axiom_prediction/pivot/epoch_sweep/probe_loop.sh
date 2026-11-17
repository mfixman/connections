#!/bin/bash
# Submits a health probe every hour; when one finishes cleanly within 15 min of starting, restarts the sweep driver and exits.
cd /home/debian/epoch-sweep
ssh_hpc() { timeout 300 ssh -F /home/debian/.ssh/config -o BatchMode=yes -o ConnectTimeout=60 hpc "$@" 2>/dev/null | grep -v X11; }
while true; do
    stamp=$(date -u +%Y%m%dT%H%M)
    out=$(python3 /home/debian/sharbaraz-monitor/submit_once.py "Sharbaraz5-15-Probe-p8-$stamp=cd /home/msf43/connections && /home/msf43/bin/icelake-batch --cpus 2 --time 0:20:00 --name Sharbaraz5-15-Probe-p8 bash /rds/user/msf43/hpc-work/epoch-stages-20261111/jobs/probe.sh" 2>&1 | tail -1)
    job=$(echo "$out" | awk '{print $2}')
    echo "$(date -u +%H:%M) probe submitted: $out" >> probe.log
    deadline=$(( $(date +%s) + 3600 ))
    verdict=""
    while [ $(date +%s) -lt $deadline ] && [ -z "$verdict" ]; do
        sleep 120
        state=$(ssh_hpc "sacct -n -X -P -j $job -o State,Elapsed | head -1")
        case "$state" in
            COMPLETED*|FAILED*|TIMEOUT*|CANCELLED*|OUT_OF*|NODE_FAIL*)
                result=$(ssh_hpc "cat \$(ls -dt ~/tmp/Sharbaraz5-15-Probe-p8-*/ | head -1)/stdout.out")
                echo "$(date -u +%H:%M) probe $job $state: $(echo "$result" | tr '\n' ' ')" >> probe.log
                if echo "$result" | grep -q "python exit 0" && echo "$result" | grep -q "PROBE DONE"; then verdict=healthy; else verdict=unhealthy; fi;;
            RUNNING*)
                elapsed=$(echo "$state" | cut -d'|' -f2)
                if [[ "$elapsed" > "00:15:00" ]]; then ssh_hpc "scancel $job"; echo "$(date -u +%H:%M) probe $job hung for 15 min; cancelled" >> probe.log; verdict=unhealthy; fi;;
        esac
    done
    if [ "$verdict" = healthy ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RESUMED: probe $job finished cleanly; driver restarted" >> sweep.log
        nohup python3 -u sweep.py >> sweep.out 2>&1 &
        echo "$(date -u +%H:%M) healthy; driver restarted" >> probe.log
        exit 0
    fi
    while [ $(date +%s) -lt $deadline ]; do sleep 60; done
done
