# Convert native SATCoP matrices and PicoSAT cores into training CSV shards.

import argparse
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "packages")]

from connections.recursion import deep_recursion
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix

from axiom_prediction.data import axiom_training_example_from_json, axiom_training_example_to_json, training_example_from_sat_core
from axiom_prediction.dataset import axiom_dataset_shard_schema, dataset_fields
from axiom_prediction.io import write_csv, write_json_atomic
from axiom_prediction.logs import log
from axiom_prediction.satcop_dump import parse_dump, read_export

export_schema = "satcop.collection.v3"
label_semantics = "native SATCoP PicoSAT clause-core membership projected to matrix clauses"

def expand_matrix(tables):
    @cache
    def term(index):
        value = tables["terms"][index]
        if value <= 0:
            return {"variable": -value}

        return {
            "symbol": value,
            "args": [
                term(tables["terms"][index + position + 1])
                for position in range(tables["symbols"][value][1])
            ],
        }

    clauses = []
    for index, row in enumerate(tables["clauses"]):
        variables, lit_start, lit_stop, *constraints, goal = row
        clause = {
            "variables": variables,
            "is_goal": bool(goal),
            "literals": [
                {"positive": bool(positive), "atom": term(atom)}
                for positive, atom in tables["literals"][lit_start:lit_stop]
            ],
        }
        if constraints:
            diseq_start, diseq_stop = constraints
            source = tables["sources"][index]
            clause["source"] = {"kind": "equality"} if source is None else {
                "kind": "input", "path": source[0], "name": source[1],
            }
            clause["disequations"] = [
                [term(left), term(right)]
                for left, right in tables["disequations"][diseq_start:diseq_stop]
            ]

        clauses.append(clause)

    return {
        "symbols": [
            {"name": name, "arity": arity, "predicate": predicate}
            for name, arity, predicate in tables["symbols"]
        ],
        "clauses": clauses,
        "have_conjecture": tables["have_conjecture"],
        **{key: tables[key] for key in ("start_clause_ids", "grounding_constant") if key in tables},
    }

def binary_identity(binary):
    with Path(binary).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def native_binary(location = None):
    path = Path(location or "~/TP/satcop").expanduser().resolve()
    binary = path / "target/release/satcop" if path.is_dir() else path
    if not binary.is_file():
        raise FileNotFoundError(f"SATCoP executable not found: {binary}; build with cargo build --release")

    result = subprocess.run([str(binary), "--help"], capture_output = True, text = True, timeout = 10)
    required = ("--collect", "--seed")
    if result.returncode or any(re.search(rf"(?m)^\s*{option}(?:\s|$)", result.stdout) is None for option in required):
        raise ValueError(f"{binary} needs the native collection export patch; see packages/axiom_prediction/README.md")

    return binary

def run_native(binary, problem, root, output, timeout_s, seed):
    environment = dict(os.environ)
    if root is not None:
        environment["TPTP_DIR"] = str(root)

    command = [
        str(binary), str(problem), "--collect", "--seed", str(seed),
        "--time", str(max(1, math.ceil(timeout_s))),
    ]
    # Native SATCoP starts one search per available CPU; index 0 uses exactly seed.
    cpus = sorted(os.sched_getaffinity(0))
    command = ["taskset", "--cpu-list", str(cpus[os.getpid() % len(cpus)]), *command]
    log_path = output.with_suffix(".log")
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

    if returncode is None:
        log_path.unlink()
        return "Timeout"

    if returncode != 0:
        raise RuntimeError(f"SATCoP exited with code {returncode}; see {log_path}")

    if status not in ("Theorem", "Unsatisfiable"):
        log_path.unlink()
        return status

    payload = parse_dump(text, problem = problem, seed = seed, timeout_s = timeout_s)
    payload.update(native_details(binary))
    write_json_atomic(output, payload)
    log_path.unlink()
    return "proved"

def run_process(command, environment, log_file, log_path, timeout_s):
    deadline = time.monotonic() + timeout_s
    process = subprocess.Popen(command, stdout = log_file, stderr = subprocess.STDOUT, env = environment)
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

def native_details(binary):
    version = subprocess.run([str(binary), "--version"], capture_output = True, text = True, check = True)
    revision = subprocess.run(
        ["git", "-C", str(Path(binary).parent), "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output = True,
        text = True,
    )
    return {
        "prover_version": version.stdout.strip(),
        "checkout_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "binary_sha256": binary_identity(binary),
    }

def stop_native(signum, frame):
    # run_process kills and reaps the Rust child when its worker is terminated.
    raise SystemExit

def collect_native_example(
    problem,
    *,
    tptp_root = None,
    timeout_s = 900,
    binary = None,
    output_dir = None,
    seed = 0,
):
    from .tptp import resolve_tptp_problem

    path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
    binary = native_binary() if binary is None else Path(binary)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix = "satcop-collection-") as directory:
            return collect_native_example(
                problem,
                tptp_root = root,
                timeout_s = timeout_s,
                binary = binary,
                output_dir = directory,
                seed = seed,
            )

    output = Path(output_dir) / (hashlib.sha256(str(problem).encode()).hexdigest() + ".json")
    output.parent.mkdir(parents = True, exist_ok = True)
    if not output.exists():
        outcome = run_native(binary, path, root, output, timeout_s, seed)
        if outcome != "proved":
            return None, outcome

    payload = json.loads(output.read_text())
    if Path(payload["problem_path"]).resolve() != path or payload["seed"] != seed:
        raise ValueError(f"SATCoP export does not match requested problem/seed: {output}")

    payload["problem_path"] = str(problem)
    example = axiom_training_example_from_json(convert_example(payload, output))
    example.graph.graph.metadata["satcop"]["binary_sha256"] = binary_identity(binary)
    # The collection cache keeps the converted example; only failures keep their native files.
    output.unlink()
    return example, "proved"

def decode_term(value, symbols, variables):
    if "variable" in value:
        return variables[value["variable"]]

    return Function(
        symbols[value["symbol"]],
        tuple(decode_term(arg, symbols, variables) for arg in value["args"]),
    )

def decode_clause(value, symbols, index):
    variables = [Variable(f"X{i}", vid = index + 1) for i in range(value["variables"])]
    literals = []
    for literal in value["literals"]:
        term = decode_term(literal["atom"], symbols, variables)
        literals.append(Literal(Atom(term.symbol, term.args), polarity = literal["positive"]))

    return Clause(
        tuple(literals),
        role = "conjecture" if value["is_goal"] else "axiom",
        free_variables = tuple(variables),
    )

@deep_recursion()
def convert_example(payload, path):
    if payload.get("schema") not in (export_schema, "satcop.collection.v2", "satcop.collection.v1"):
        raise ValueError(f"unsupported SATCoP export schema in {path}")

    if payload.get("status") not in ("Theorem", "Unsatisfiable"):
        raise ValueError(f"SATCoP export has no successful refutation: {path}")

    native = payload["matrix"]
    if payload["schema"] != "satcop.collection.v1":
        native = expand_matrix(native)
    # Native IDs distinguish generated and input symbols even if their names collide.
    symbols = [f"{index}:{symbol['name']}" for index, symbol in enumerate(native["symbols"])]
    clauses = tuple(
        decode_clause(clause, symbols, index)
        for index, clause in enumerate(native["clauses"])
    )

    matrix = Matrix(clauses, source_has_conjecture = native["have_conjecture"])
    conjectures = list(matrix.conjecture_clauses)
    axioms = [index for index, clause in enumerate(clauses) if clause.role == "axiom"]
    core = payload["core_clause_ids"]
    if not core or not set(core) <= set(range(len(clauses))):
        raise ValueError(f"empty or out-of-matrix SATCoP core in {path}")

    example = training_example_from_sat_core(
        matrix,
        problem_path = payload["problem_path"],
        axiom_clause_ids = axioms,
        conjecture_clause_ids = conjectures,
        core_clause_ids = core,
    )

    provenance = {
        key: value for key, value in payload.items()
        if key not in {"matrix", "problem_path"}
    }
    provenance.update(
        export_path = str(path.resolve()),
        matrix_sha256 = hashlib.sha256(
            json.dumps(native, sort_keys = True, separators = (",", ":")).encode()
        ).hexdigest(),
    )
    if "start_clause_ids" in native:
        provenance.update(start_clause_ids = native["start_clause_ids"])

    if "source" in native["clauses"][0]:
        provenance.update(clause_sources = [clause["source"] for clause in native["clauses"]])
        provenance["disequations"] = {
            str(index): clause["disequations"]
            for index, clause in enumerate(native["clauses"])
            if clause["disequations"]
        }

    example.graph.graph.metadata["satcop"] = provenance
    return axiom_training_example_to_json(example)

def convert_exports(paths, output, **metadata):
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing dataset shard: {output}")

    collection = None
    count = 0

    def records():
        nonlocal collection, count
        for path in paths:
            payload = read_export(path, **metadata)
            row = convert_example(payload, path)
            settings = {
                "sat_policy": "satcop",
                "prover_backend": "rust-satcop",
                "prover_version": payload["prover_version"],
                "prover_revision": payload.get("prover_revision"),
                "export_schema": payload["schema"],
                "label_semantics": label_semantics,
            }

            if collection is None:
                collection = settings
                yield {
                    "schema": axiom_dataset_shard_schema,
                    "record": "metadata",
                    "collection": collection,
                }
            elif settings != collection:
                raise ValueError(f"incompatible SATCoP versions in {path}; use separate datasets")

            yield row
            count += 1
            problem = row["problem_path"]
            log(f"converted {problem}: {int(sum(row['labels']))}/{len(row['labels'])} core axioms")

        if count == 0:
            raise ValueError("no SATCoP exports supplied")

    output.parent.mkdir(parents = True, exist_ok = True)
    write_csv(output, records(), dataset_fields)
    return count

def main():
    parser = argparse.ArgumentParser(description = "Convert native SATCoP exports to a training CSV")
    parser.add_argument("exports", nargs = "+", type = Path, help = "saved JSON exports, raw SATCoP stdout files, or flat directories")
    parser.add_argument("--output", required = True, type = Path, help = "new training CSV shard")
    parser.add_argument("--problem", help = "original input path for a single raw stdout dump")
    parser.add_argument("--seed", type = int, help = "seed used for a raw stdout dump")
    parser.add_argument("--timeout-seconds", type = float, help = "budget used for a raw stdout dump")
    args = parser.parse_args()

    if args.output.suffix != ".csv":
        parser.error("--output must end in .csv")

    paths = []
    for path in args.exports:
        paths.extend((sorted(path.glob("*.json")) or sorted(path.glob("*.out"))) if path.is_dir() else [path])

    if args.problem is not None and len(paths) != 1:
        parser.error("--problem requires exactly one export")

    count = convert_exports(paths, args.output, problem = args.problem, seed = args.seed, timeout_s = args.timeout_seconds)
    log(f"wrote {count} examples to {args.output}")

if __name__ == "__main__":
    main()
