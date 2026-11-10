# Runs an unmodified external prover on Connections' clauses and records which input clauses each proof used.
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

checkout = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(checkout / "packages"), str(checkout / "src")]

from axiom_prediction.prover_cores.provers import Prover, modules

def run_prover(command, timeout_s):
    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, start_new_session = True)
    try:
        return process.communicate(timeout = timeout_s)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()[0]

def example_json(problem, used):
    from axiom_prediction.data import axiom_training_example_to_json, training_example_from_sat_core

    example = training_example_from_sat_core(
        problem.matrix,
        problem_path = str(problem.path),
        axiom_clause_ids = list(problem.axiom_clause_ids),
        conjecture_clause_ids = list(problem.conjecture_clause_ids),
        core_clause_ids = used,
    )

    return axiom_training_example_to_json(example)

def collect(task):
    path, prover, seconds, cpu, scratch, tptp = task
    from axiom_prediction.tptp import load_tptp_problem

    os.sched_setaffinity(0, {cpu})
    module = modules[prover]
    started = time.monotonic()
    row = {"problem": Path(path).name, "problem_path": path, "proved": False, "used": []}
    prover_input = Path(scratch) / f"{Path(path).stem}-{cpu}.in"
    try:
        problem = load_tptp_problem(path, tptp_root = tptp)
        module.write_input(problem.matrix, set(problem.conjecture_clause_ids), prover_input, seconds)
        output = run_prover(module.command(prover_input, seconds), seconds + 5)
        row["proved"] = module.proved(output)
        row["used"] = module.used_indices(output) if row["proved"] else []
        if row["used"]:
            row["example"] = example_json(problem, row["used"])
    except Exception as error:
        row["error"] = repr(error)[:200]

    prover_input.unlink(missing_ok = True)
    row["seconds"] = round(time.monotonic() - started, 2)
    return row

def refutable_problems(args):
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.tptp import declared_non_refutable, tptp_problems

    split = ProblemSplit(split = args.split, parts = args.parts, scheme = 5)
    problems = split.select(tptp_problems(tptp_root = args.tptp))
    return [path for path in problems if declared_non_refutable(path) is None]

def parse_arguments():
    parser = argparse.ArgumentParser(description = "collect the input clauses used in an external prover's proofs")
    parser.add_argument("--prover", type = Prover, choices = list(Prover), required = True)
    parser.add_argument("--output", type = Path, required = True, help = "JSON lines, resumed when it exists")
    parser.add_argument("--tptp", default = "/rds/user/msf43/hpc-work/TPTP")
    parser.add_argument("--split", type = int, default = 10)
    parser.add_argument("--parts", type = int, nargs = "+", default = list(range(10)))
    parser.add_argument("--problems", nargs = "+", help = "instead of the split's problems")
    parser.add_argument("--seconds", type = float, default = 60)
    parser.add_argument("--workers", type = int, default = 76)
    parser.add_argument("--scratch", default = "/dev/shm/prover-cores")
    return parser.parse_args()

def main():
    args = parse_arguments()
    Path(args.scratch).mkdir(parents = True, exist_ok = True)
    args.output.parent.mkdir(parents = True, exist_ok = True)
    done = {json.loads(line)["problem"] for line in args.output.open()} if args.output.exists() else set()
    paths = [path for path in args.problems or refutable_problems(args) if Path(path).name not in done]

    cpus = sorted(os.sched_getaffinity(0))
    tasks = [
        (path, args.prover, args.seconds, cpus[index % len(cpus)], args.scratch, args.tptp)
        for index, path in enumerate(paths)
    ]

    print(f"{args.prover}: {len(paths)} problems to do, {len(done)} done, {len(cpus)} cpus", flush = True)
    context = multiprocessing.get_context("spawn")
    counts = {"finished": 0, "proved": 0, "errors": 0}
    with args.output.open("a") as stream, ProcessPoolExecutor(min(args.workers, len(cpus)), mp_context = context) as pool:
        for future in as_completed([pool.submit(collect, task) for task in tasks]):
            row = future.result()
            stream.write(json.dumps(row) + "\n")
            stream.flush()

            counts["finished"] += 1
            counts["proved"] += row["proved"]
            counts["errors"] += "error" in row
            if counts["finished"] % 200 == 0:
                print(counts, flush = True)

    print(f"finished: {counts}", flush = True)

if __name__ == "__main__":
    main()
