# Native SATCoP proof search for run, optionally ordering extensions by clause weights.

from pathlib import Path
import shutil
import tempfile
import time

from connections.recursion import deep_recursion

from .choices import GuidanceMode
from .guided import clause_weights
from .limits import CollectionTimeout
from .satcop import axiom_clause_ids, decode_matrix, expand_matrix, native_binary, run_logged
from .satcop_dump import parse_matrix

def search_native(path, root, config, predictor, started):
    directory = Path(tempfile.mkdtemp(prefix = "satcop-run-"))
    try:
        out = search_in(directory, path, root, config, predictor, started)
    except CollectionTimeout:
        shutil.rmtree(directory)
        raise

    # Like collection, only failures keep their native files.
    shutil.rmtree(directory)
    return out

def search_binary():
    return native_binary(options = ("--collect", "--seed", "--weights"))

def search_in(directory, path, root, config, predictor, started):
    binary = search_binary()
    deadline = started + config.timeout_s
    options = ["--seed", str(config.seed), "--quiet"]
    out = {}
    if config.mode != GuidanceMode.Base:
        weights, axioms = native_weights(binary, path, root, config, predictor, directory, deadline)
        (directory / "weights").write_text("\n".join(map(str, weights)) + "\n")
        options += ["--weights", str(directory / "weights")]
        out.update(axioms = axioms, prediction_seconds = time.monotonic() - started)

    status = run_satcop(binary, path, options, root, directory / "search.log", deadline)[0]
    out.update(
        outcome = status,
        proved = status in ("Theorem", "Unsatisfiable"),
        seconds = time.monotonic() - started,
    )

    return out

# Scores SATCoP's own clausification, which the guided search reproduces clause for clause.
@deep_recursion()
def native_weights(binary, path, root, config, predictor, directory, deadline):
    min_weight = 1e-6

    options = ["--collect", "--clausify"]
    text = run_satcop(binary, path, options, root, directory / "matrix.log", deadline)[1]
    matrix = decode_matrix(expand_matrix(parse_matrix(text)))
    axioms = axiom_clause_ids(matrix)
    scores = clause_weights(matrix, axioms, list(matrix.conjecture_clauses), predictor)

    # SATCoP samples in proportion to w; w ** (1 / T) matches AxiomGuided's temperature.
    weights = [
        max(min_weight, scores[index]) ** (1 / config.temperature)
        for index in range(len(matrix.clauses))
    ]

    return weights, len(axioms)

def run_satcop(binary, path, options, root, log_path, deadline):
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise CollectionTimeout

    returncode, status, text = run_logged(binary, path, options, root, log_path, remaining_s)
    if returncode is None:
        raise CollectionTimeout

    if returncode != 0:
        lines = text.strip().splitlines()
        raise RuntimeError(f"SATCoP exited with code {returncode}: {lines[-1] if lines else ''}")

    return status, text
