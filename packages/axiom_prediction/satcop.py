# Convert native SATCoP matrices and PicoSAT cores into training CSV shards.

import argparse
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

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "packages")]

from connections.recursion import deep_recursion
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix

from axiom_prediction.data import axiom_training_example_from_json, axiom_training_example_to_json, training_example_from_sat_core
from axiom_prediction.dataset import axiom_dataset_shard_schema, dataset_fields
from axiom_prediction.io import write_csv
from axiom_prediction.logs import log

export_schema = "satcop.collection.v1"
label_semantics = "native SATCoP PicoSAT clause-core membership projected to matrix clauses"

def binary_identity(binary):
    with Path(binary).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def native_binary(location = None):
    path = Path(location or "~/TP/satcop").expanduser().resolve()
    binary = path / "target/release/satcop" if path.is_dir() else path
    if not binary.is_file():
        raise FileNotFoundError(f"SATCoP executable not found: {binary}; build with cargo build --release")

    result = subprocess.run([str(binary), "--help"], capture_output = True, text = True, timeout = 10)
    required = ("--collection-output", "--threads", "--seed")
    if result.returncode or any(option not in result.stdout for option in required):
        raise ValueError(f"{binary} needs the native collection export patch; see packages/axiom_prediction/README.md")

    return binary

def run_native(binary, problem, root, output, timeout_s, seed):
    environment = dict(os.environ)
    if root is not None:
        environment["TPTP_DIR"] = str(root)

    with tempfile.TemporaryDirectory(prefix = ".satcop-", dir = output.parent) as temporary:
        export = Path(temporary) / "collection.json"
        command = [
            str(binary), str(problem), "--quiet", "--threads", "1",
            "--seed", str(seed), "--time", str(max(1, math.ceil(timeout_s))),
            "--collection-output", str(export),
        ]

        with output.with_suffix(".log").open("w") as log_file:
            log_file.write(json.dumps(command) + "\n")
            log_file.flush()
            previous = signal.signal(signal.SIGTERM, stop_native)
            try:
                result = subprocess.run(
                    command,
                    stdout = log_file,
                    stderr = subprocess.STDOUT,
                    env = environment,
                    timeout = timeout_s,
                )
            except subprocess.TimeoutExpired:
                return "Timeout"
            finally:
                signal.signal(signal.SIGTERM, previous)

        statuses = re.findall(r"% SZS status (\w+)", output.with_suffix(".log").read_text())
        if statuses and statuses[-1] == "SyntaxError":
            from connections.parsing.tptp import TPTPParseError

            raise TPTPParseError("SATCoP", f"native parse error; see {output.with_suffix('.log')}", 0, 0)

        if result.returncode:
            raise RuntimeError(f"SATCoP exited with code {result.returncode}; see {output.with_suffix('.log')}")

        if export.exists():
            export.replace(output)
            return "proved"

    if statuses and statuses[-1] in ("Theorem", "Unsatisfiable"):
        raise RuntimeError(f"SATCoP reported success without exporting its core: {output}")

    return statuses[-1] if statuses else "Unknown"

def stop_native(signum, frame):
    # subprocess.run kills and reaps the Rust child when its worker is terminated.
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
    if payload.get("schema") != export_schema:
        raise ValueError(f"unsupported SATCoP export schema in {path}")

    if payload.get("status") not in ("Theorem", "Unsatisfiable"):
        raise ValueError(f"SATCoP export has no successful refutation: {path}")

    native = payload["matrix"]
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
        start_clause_ids = native["start_clause_ids"],
        clause_sources = [clause["source"] for clause in native["clauses"]],
        disequations = {
            str(index): clause["disequations"]
            for index, clause in enumerate(native["clauses"])
            if clause["disequations"]
        },
    )

    example.graph.graph.metadata["satcop"] = provenance
    return axiom_training_example_to_json(example)

def convert_exports(paths, output):
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing dataset shard: {output}")

    seen = set()
    collection = None
    count = 0

    def records():
        nonlocal collection, count
        for path in paths:
            payload = json.loads(path.read_text())
            row = convert_example(payload, path)
            settings = {
                "sat_policy": "satcop",
                "prover_backend": "rust-satcop",
                "prover_version": payload["prover_version"],
                "prover_revision": payload.get("prover_revision"),
                "export_schema": export_schema,
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

            problem = row["problem_path"]
            if problem in seen:
                raise ValueError(f"multiple exports for {problem}; the training loader keeps one per problem")

            seen.add(problem)
            yield row
            count += 1
            log(f"converted {problem}: {int(sum(row['labels']))}/{len(row['labels'])} core axioms")

        if count == 0:
            raise ValueError("no SATCoP exports supplied")

    output.parent.mkdir(parents = True, exist_ok = True)
    write_csv(output, records(), dataset_fields)
    return count

def main():
    parser = argparse.ArgumentParser(description = "Convert native SATCoP exports to a training CSV")
    parser.add_argument("exports", nargs = "+", type = Path, help = "export JSON files or flat directories")
    parser.add_argument("--output", required = True, type = Path, help = "new training CSV shard")
    args = parser.parse_args()

    if args.output.suffix != ".csv":
        parser.error("--output must end in .csv")

    paths = []
    for path in args.exports:
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])

    count = convert_exports(paths, args.output)
    log(f"wrote {count} examples to {args.output}")

if __name__ == "__main__":
    main()
