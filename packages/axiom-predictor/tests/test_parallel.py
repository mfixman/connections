from __future__ import annotations

import axiom_prediction.parallel as parallel


def test_explicit_worker_count_is_capped_by_tasks():
    assert parallel.determine_worker_count(3, 20) == 3
    assert parallel.determine_worker_count(None, 20) == 20


def test_auto_worker_count_respects_slurm_and_affinity(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setattr(parallel, "_process_cpu_count", lambda: 32)
    assert parallel.determine_worker_count(100) == 8


def test_auto_worker_count_uses_available_cpus_without_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(parallel, "_process_cpu_count", lambda: 24)
    assert parallel.determine_worker_count(100) == 24


def test_empty_work_has_no_workers():
    assert parallel.determine_worker_count(0, 8) == 0
