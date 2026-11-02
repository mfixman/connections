import os
import subprocess
import sys

from axiom_prediction.memory_guard import enforce_limit, slurm_memory_limit_b

def sleeper(size_mib):
    code = f"import time; data = bytearray({size_mib} << 20); print(flush = True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", code], stdout = subprocess.PIPE)
    process.stdout.readline()
    return process

def test_guard_kills_only_the_largest_descendant_over_the_limit():
    small, large = sleeper(1), sleeper(200)
    try:
        assert not enforce_limit(os.getpid(), 1 << 50)
        assert small.poll() is None and large.poll() is None
        assert enforce_limit(os.getpid(), 1)
        assert large.wait(timeout = 10) == -9
        assert small.poll() is None
    finally:
        small.kill()
        large.kill()
        small.communicate()
        large.communicate()

def test_guard_is_off_outside_slurm(monkeypatch):
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(name, raising = False)

    assert slurm_memory_limit_b() is None
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "3370")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "38")
    assert slurm_memory_limit_b() == 3370 * 38 << 20
