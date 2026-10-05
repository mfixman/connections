from __future__ import annotations

import os

def determine_worker_count(
    total_tasks: int | None,
    requested_workers: int | None = None,
) -> int:
    if total_tasks is not None and total_tasks <= 0:
        return 0

    if requested_workers is not None:
        if requested_workers < 1:
            raise ValueError("num_workers must be at least 1")

        workers = requested_workers
    else:
        candidates = [
            count
            for count in (slurm_cpus_per_task(), process_cpu_count())
            if count is not None
        ]

        workers = min(candidates) if candidates else 1

    if total_tasks is not None:
        workers = min(workers, total_tasks)

    return max(1, workers)

def slurm_cpus_per_task() -> int | None:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return None

    try:
        value = int(raw)
    except ValueError:
        return None

    return value if value > 0 else None

def process_cpu_count() -> int | None:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count()

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
