# Slurm marks a job OUT_OF_MEMORY after any kernel OOM kill, so the largest search dies first.

import os
from pathlib import Path
import signal
import threading

from .logs import log

def slurm_memory_limit_b():
    b_per_mib = 1 << 20
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        return int(per_node) * b_per_mib

    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    return int(per_cpu) * int(cpus) * b_per_mib if per_cpu and cpus else None

def process_rss_b(pid):
    page_size_b = os.sysconf("SC_PAGE_SIZE")
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * page_size_b
    except (OSError, IndexError, ValueError):
        return 0

def child_pids(pid):
    try:
        return list(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
    except OSError:
        return []

def descendants(pid):
    children = child_pids(pid)
    return children + [grandchild for child in children for grandchild in descendants(child)]

def enforce_limit(root, limit_b):
    sizes = {pid: process_rss_b(pid) for pid in descendants(root)}
    total_b = process_rss_b(root) + sum(sizes.values())
    if total_b < limit_b or not sizes:
        return False

    victim = max(sizes, key = sizes.get)
    try:
        os.kill(victim, signal.SIGKILL)
    except ProcessLookupError:
        return False

    b_per_gib = 1 << 30
    log(f"memory guard: killed {victim} ({sizes[victim] / b_per_gib:.1f}G of {total_b / b_per_gib:.1f}G)")
    return True

def guard_loop(stop, root, limit_b):
    poll_interval_s = 0.25
    # An exiting process keeps its RSS until it has released it.
    settle_s = 2.0
    while not stop.wait(poll_interval_s):
        if enforce_limit(root, limit_b):
            stop.wait(settle_s)

def start_memory_guard():
    usable_fraction = 0.9
    limit_b = slurm_memory_limit_b()
    stop = threading.Event()
    if limit_b is None:
        return stop

    arguments = (stop, os.getpid(), limit_b * usable_fraction)
    threading.Thread(target = guard_loop, args = arguments, daemon = True).start()
    return stop
