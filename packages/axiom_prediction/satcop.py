# Runs native (Rust) SATCoP and labels its matrix with the PicoSAT core of the proof.

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .data import training_example_from_sat_core
from .satcop_dump import axiom_clause_ids, parse_proof, proved_statuses

label_semantics = "native SATCoP PicoSAT clause-core membership projected to matrix clauses"

def binary_identity(binary):
    with Path(binary).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def native_binary(location = None, options = ("--collect", "--seed")):
    path = Path(location or "~/TP/satcop").expanduser().resolve()
    binary = path / "target/release/satcop" if path.is_dir() else path
    if not binary.is_file():
        raise FileNotFoundError(f"SATCoP executable not found: {binary}; build with cargo build --release")

    result = subprocess.run([str(binary), "--help"], capture_output = True, text = True, timeout = 10)
    if result.returncode or any(re.search(rf"(?m)^\s*{option}(?:\s|$)", result.stdout) is None for option in options):
        raise ValueError(f"{binary} lacks {' '.join(options)}; build the axiom_selection branch, see packages/axiom_prediction/README.md")

    return binary

# Returns (exit code or None on timeout, last SZS status, output); panics and parse errors raise.
def run_logged(binary, problem, options, root, log_path, timeout_s):
    environment = dict(os.environ)
    if root is not None:
        environment["TPTP_DIR"] = str(root)

    # Native SATCoP starts one search per available CPU; index 0 uses exactly seed.
    cpus = sorted(os.sched_getaffinity(0))
    cpu = str(cpus[os.getpid() % len(cpus)])
    command = ["taskset", "--cpu-list", cpu, str(binary), str(problem), *options]
    with log_path.open("w") as log_file:
        log_file.write(json.dumps(command) + "\n")
        log_file.flush()
        previous = signal.signal(signal.SIGTERM, stop_native)
        try:
            returncode = run_process(command, environment, log_file, log_path, timeout_s)
        finally:
            signal.signal(signal.SIGTERM, previous)

    text = log_path.read_text()
    if re.search(r"^thread .*panicked at", text, re.MULTILINE):
        raise RuntimeError(f"SATCoP search panicked; see {log_path}")

    statuses = re.findall(r"% SZS status (\w+)", text)
    status = statuses[-1] if statuses else "Unknown"
    if status == "SyntaxError":
        from connections.parsing.tptp import TPTPParseError

        raise TPTPParseError("SATCoP", f"native parse error; see {log_path}", 0, 0)

    return returncode, status, text

# A worker killed before it can reap SATCoP would otherwise leave it running, stealing CPU from later searches.
# The library is loaded before forking, since loading it in the child can deadlock.
libc = ctypes.CDLL(None)

def die_with_parent():
    pr_set_pdeathsig = 1
    libc.prctl(pr_set_pdeathsig, signal.SIGKILL)

def run_process(command, environment, log_file, log_path, timeout_s):
    deadline = time.monotonic() + timeout_s
    process = subprocess.Popen(
        command,
        stdout = log_file,
        stderr = subprocess.STDOUT,
        env = environment,
        preexec_fn = die_with_parent,
    )

    try:
        with log_path.open() as monitor:
            while process.poll() is None:
                if re.search(r"^thread .*panicked at", monitor.read(), re.MULTILINE):
                    raise RuntimeError(f"SATCoP search panicked; see {log_path}")

                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return None

                try:
                    return process.wait(timeout = min(0.2, remaining_s))
                except subprocess.TimeoutExpired:
                    pass

        return process.returncode
    finally:
        if process.poll() is None:
            process.kill()

        process.wait()

def stop_native(signum, frame):
    # run_process kills and reaps the Rust child when its worker is terminated.
    raise SystemExit

# Only runs that crash keep their log in output_dir.
def collect_native_example(problem, *, tptp_root, timeout_s, binary, output_dir, seed):
    from .tptp import resolve_tptp_problem

    path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
    log_path = Path(output_dir) / (hashlib.sha256(str(problem).encode()).hexdigest() + ".log")
    log_path.parent.mkdir(parents = True, exist_ok = True)
    options = ["--collect", "--seed", str(seed), "--time", str(max(1, math.ceil(timeout_s)))]
    returncode, status, text = run_logged(Path(binary), path, options, root, log_path, timeout_s)
    if returncode not in (None, 0):
        raise RuntimeError(f"SATCoP exited with code {returncode}; see {log_path}")

    if returncode is None or status not in proved_statuses:
        log_path.unlink()
        return None, "Timeout" if returncode is None else status

    matrix, core = parse_proof(text)
    example = training_example_from_sat_core(
        matrix,
        problem_path = str(problem),
        axiom_clause_ids = axiom_clause_ids(matrix),
        conjecture_clause_ids = list(matrix.conjecture_clauses),
        core_clause_ids = core,
    )

    log_path.unlink()
    return example, "proved"
