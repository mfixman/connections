# Runs an unmodified external prover on Connections' clauses and records which input clauses each proof used.
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

checkout = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(checkout / "packages"), str(checkout / "src")]

from axiom_prediction.prover_cores.provers import Prover, modules
from axiom_prediction.satcop import die_with_parent
from axiom_prediction.search_workers import supervised_results

def run_prover(command, timeout_s):
    process = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        start_new_session = True,
        preexec_fn = die_with_parent,
    )

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

# Runs in a supervised worker, which turns an exception into an error row.
def collect(path, prover, seconds, tptp, max_example_mb):
    from axiom_prediction.tptp import load_tptp_problem

    module = modules[prover]
    problem = load_tptp_problem(path, tptp_root = tptp)
    fof = "+" in Path(path).name
    with tempfile.NamedTemporaryFile(suffix = ".in") as prover_input:
        module.write_input(problem.matrix, set(problem.conjecture_clause_ids), fof, prover_input.name, seconds)
        output = run_prover(module.command(prover_input.name, seconds), seconds + 5)

    row = {"problem": path, "proved": module.proved(output), "used": []}
    if not row["proved"]:
        return row

    row["used"] = module.used_indices(output)
    if not row["used"]:
        return row

    # Training drops examples above this size, and the largest ones take gigabytes to build and store.
    example = example_json(problem, row["used"])
    if len(json.dumps(example)) > max_example_mb * 1e6:
        return {**row, "outcome": "ExampleTooLarge"}

    return {**row, "example": example}

# Rows made by the supervisor (exception, timeout, dead worker) carry only an outcome.
def output_row(result):
    path = result.pop("problem")
    row = {"problem": Path(path).name, "problem_path": path, "used": [], **result}
    row["seconds"] = round(row["seconds"], 2)
    if "outcome" in row:
        row["error"] = row.pop("outcome")

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
    parser.add_argument("--scratch", default = "/dev/shm", help = "parent of the workers' temporary directories")
    parser.add_argument("--max-example-mb", type = float, default = 10)
    return parser.parse_args()

def main():
    # Parsing and clausifying a problem gets this long on top of the prover's time.
    load_limit_s = 120

    args = parse_arguments()
    tempfile.tempdir = args.scratch
    args.output.parent.mkdir(parents = True, exist_ok = True)
    done = {json.loads(line)["problem"] for line in args.output.open()} if args.output.exists() else set()
    paths = [path for path in args.problems or refutable_problems(args) if Path(path).name not in done]
    tasks = [(path, args.prover, args.seconds, args.tptp, args.max_example_mb) for path in paths]
    workers = min(args.workers, len(os.sched_getaffinity(0)))
    print(f"{args.prover}: {len(paths)} problems to do, {len(done)} done, {workers} workers", flush = True)

    counts = {"finished": 0, "proved": 0, "errors": 0}
    results = supervised_results(collect, tasks, workers = workers, timeout = args.seconds + load_limit_s)
    with args.output.open("a") as stream:
        for row in map(output_row, results):
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
