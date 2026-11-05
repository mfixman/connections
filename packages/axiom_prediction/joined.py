# Joins collect results into one deduplicated file of tight numerical arrays.

from collections import defaultdict
from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np

from axiom_prediction.io import sized_csv_rows, tmp_path, write_json_atomic
from axiom_prediction.representation.schema import GraphInput

from .data import AxiomTrainingExample, axiom_example_schema
from .dataset import axiom_dataset_schema, axiom_dataset_shard_schema, merged_collection, paused_gc, read_object
from .graph import AxiomGraph
from .logs import log
from .logs import progress as track_progress
from .parallel import determine_worker_count
from .problems_file import problem_name, problem_schema

joined_schema = "learncop.axiom_prediction.joined.v1"

table_widths = {
    "nodes.symbol": 2,
    "nodes.clause": 4,
    "nodes.literal": 1,
    "nodes.term": 1,
    "nodes.var": 1,

    "edges.contains": 2,
    "edges.atom": 2,
    "edges.lit_sym": 2,

    "edges.sym": 2,
    "edges.arg_term": 3,
    "edges.arg_var": 3,

    "edges.complement": 2,
}

@dataclass(frozen = True, slots = True)
class CompactGraph:
    name: str
    digest: bytes
    tables: dict[str, np.ndarray]
    core: tuple[int, ...]
    record_bytes: int

@dataclass(frozen = True, slots = True)
class JoinInput:
    path: str
    collection: dict
    graphs: list[CompactGraph]
    failures: dict[str, str]

def input_files(path):
    path = Path(path)
    if path.is_file():
        found = [path]
    else:
        # Old layouts keep full results in the cache; their problems.csv may be a stub.
        found = sorted(path.glob(".cache/*/cache.csv")) or sorted(path.glob("*.csv"))

    if not found:
        raise FileNotFoundError(f"no collect results in {path}")

    if any(file.suffix != ".csv" for file in found):
        raise ValueError(f"join reads CSV collect results, not {path}")

    return [file.resolve() for file in found]

def one_way_complements(rows, name):
    # Edges come in mirrored pairs; storing one side halves the largest table.
    scale = int(rows.max(initial = 0)) + 1
    forward = rows[:, 0].astype(np.int64) * scale + rows[:, 1]
    backward = rows[:, 1].astype(np.int64) * scale + rows[:, 0]
    if (rows[:, 0] == rows[:, 1]).any() or not np.array_equal(np.sort(forward), np.sort(backward)):
        raise ValueError(f"complement edges of {name} are not mirrored pairs")

    return rows[rows[:, 0] < rows[:, 1]]

def compact_row(name, row, record_bytes):
    graph = row["graph"]
    if row["schema"] == axiom_example_schema:
        if (graph["preprocessor"], graph["version"]) != ("axiom_prediction", "1"):
            raise ValueError(f"unsupported axiom graph format in {name}")

    tables = {}
    for table, width in table_widths.items():
        kind, relation = table.split(".")
        rows = graph[kind].get(relation, [])
        tables[table] = np.asarray(rows, dtype = np.int32).reshape(-1, width)

    tables["edges.complement"] = one_way_complements(tables["edges.complement"], name)
    tables["axiom_ids"] = np.asarray(row["axiom_clause_ids"], dtype = np.int32)
    tables["conjecture_ids"] = np.asarray(row["conjecture_clause_ids"], dtype = np.int32)
    if len(row["labels"]) != len(tables["axiom_ids"]):
        raise ValueError(f"label count must match axiom clause count in {name}")

    digest = hashlib.blake2b(digest_size = 16)
    for table, values in tables.items():
        digest.update(table.encode())
        digest.update(values.tobytes())

    core = tuple(index for index, label in enumerate(row["labels"]) if label)
    return CompactGraph(name, digest.digest(), tables, core, record_bytes)

def read_input(path):
    known_schemas = {problem_schema, axiom_example_schema, axiom_dataset_schema, axiom_dataset_shard_schema}

    graphs, failures = [], {}
    collection = None
    for record_bytes, row in sized_csv_rows(path):
        if row.get("schema") not in known_schemas:
            raise ValueError(f"unsupported schema {row.get('schema')!r} in {path}")

        collection = collection or row.get("collection")
        if row.get("record") == "metadata":
            continue

        name = problem_name(row.get("problem_name") or row.get("problem_path") or row["problem"])
        if "graph" in row:
            graphs.append(compact_row(name, row, record_bytes))
        else:
            failures[name] = str(row.get("outcome", "failed"))

    if not graphs and not failures:
        raise ValueError(f"{path} has no results")

    return JoinInput(str(path), collection or {}, graphs, failures)

def read_inputs(files, num_workers):
    workers = determine_worker_count(len(files), num_workers)
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer = gc.disable) as pool:
        results = pool.imap_unordered(read_input, files)
        yield from track_progress(results, "reading collect results", len(files))

def source_summary(result):
    return {
        "path": result.path,
        "collection": result.collection,
        "proved": len(result.graphs),
        "failed": len(result.failures),
    }

def sources_overview(sources):
    collections = [source["collection"] for source in sources]
    parts = {part for source in sources for part in re.findall(r"part-(\d+)", source["path"])}
    return {
        "provers": sorted({collection.get("sat_policy", "") for collection in collections}),
        "seeds": sorted({collection.get("seed", 0) for collection in collections}),
        "parts": sorted(map(int, parts)),
    }

def narrow(array):
    if array.size == 0 or array.min() < 0:
        return array

    return array.astype(np.min_scalar_type(int(array.max())))

def add_table(arrays, name, rows):
    arrays[name] = narrow(np.concatenate(rows))
    arrays[f"{name}.counts"] = np.asarray(list(map(len, rows)), dtype = np.int64)

def stored_arrays(graphs, cores, sizes, failures):
    keys = sorted(graphs)
    examples = [
        (index, core)
        for index, key in enumerate(keys)
        for core in sorted(cores[key])
    ]

    arrays = {
        "problem_names": np.asarray([name for name, _ in keys]),
        "record_bytes": np.asarray([sizes[key] for key in keys], dtype = np.int64),
        "example_graphs": np.asarray([index for index, _ in examples], dtype = np.int64),
        "failures": np.frombuffer(json.dumps(failures, sort_keys = True).encode(), dtype = np.uint8),
    }

    for table in (*table_widths, "axiom_ids", "conjecture_ids"):
        add_table(arrays, table, [graphs[key].tables[table] for key in keys])

    add_table(arrays, "core_positions", [np.asarray(core, dtype = np.int32) for _, core in examples])
    return arrays

def write_arrays(path, arrays):
    tmp = tmp_path(path)
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok = True)

def join_datasets(inputs, output_dir, num_workers = None):
    output = Path(output_dir)
    data, description = output / "data.npz", output / "metadata.json"
    if data.exists() or description.exists():
        raise FileExistsError(f"refusing to replace the joined dataset in {output}")

    files = list(dict.fromkeys(file for path in inputs for file in input_files(path)))
    graphs, sizes, cores = {}, {}, defaultdict(set)
    failures, sources = {}, []
    for result in read_inputs(files, num_workers):
        sources.append(source_summary(result))
        for graph in result.graphs:
            key = (graph.name, graph.digest)
            graphs.setdefault(key, graph)
            sizes[key] = min(sizes.get(key, graph.record_bytes), graph.record_bytes)
            cores[key].add(graph.core)

        for name, outcome in result.failures.items():
            failures[name] = min(outcome, failures.get(name, outcome))

    if not graphs:
        raise RuntimeError(f"no proved problems in the {len(files)} inputs")

    proved = {name for name, _ in graphs}
    failures = {name: outcome for name, outcome in failures.items() if name not in proved}
    sources.sort(key = lambda source: source["path"])

    examples = sum(map(len, cores.values()))
    counts = {"problems": len(proved), "graphs": len(graphs), "examples": examples, "failures": len(failures)}
    log(f"joined {len(files)} inputs: {counts}; writing {data}")

    output.mkdir(parents = True, exist_ok = True)
    write_arrays(data, stored_arrays(graphs, cores, sizes, failures))
    write_json_atomic(description, {
        "schema": joined_schema,
        "collection": merged_collection({source["path"]: source["collection"] for source in sources}),
        **sources_overview(sources),
        **counts,
        "inputs": sources,
    })

    return {"inputs": len(files), **counts, "data_bytes": data.stat().st_size, "dataset": str(output)}

def kept_rows(arrays, name, keep):
    counts = arrays[f"{name}.counts"]
    rows = arrays[name] if keep.all() else arrays[name][np.repeat(keep, counts)]
    return np.split(rows, np.cumsum(counts[keep])[:-1])

def stored_graphs(arrays, keep):
    tables = {name: kept_rows(arrays, name, keep) for name in table_widths}
    axioms = kept_rows(arrays, "axiom_ids", keep)
    conjectures = kept_rows(arrays, "conjecture_ids", keep)

    graphs = {}
    for position, index in enumerate(np.flatnonzero(keep).tolist()):
        nodes, edges = {}, {}
        for table in table_widths:
            kind, relation = table.split(".")
            (nodes if kind == "nodes" else edges)[relation] = tables[table][position]

        edges["complement"] = np.concatenate((edges["complement"], edges["complement"][:, ::-1]))
        graph = GraphInput(nodes, edges, [], "axiom_prediction", "1")
        graphs[index] = AxiomGraph(graph, axioms[position].tolist(), conjectures[position].tolist())

    return graphs

def stored_examples(arrays, graphs):
    names = arrays["problem_names"].tolist()
    every = np.ones(len(arrays["example_graphs"]), dtype = bool)
    cores = kept_rows(arrays, "core_positions", every)
    for index, core in zip(arrays["example_graphs"].tolist(), cores, strict = True):
        graph = graphs.get(index)
        if graph is None:
            continue

        labels = [0.0] * len(graph.axiom_clause_ids)
        for position in core.tolist():
            labels[position] = 1.0

        yield AxiomTrainingExample(
            problem_path = names[index],
            matrix = None,
            graph = graph,
            labels = labels,
        )

@paused_gc()
def load_joined(directory, max_bytes = None):
    metadata = read_object(directory / "metadata.json")
    if metadata.get("schema") != joined_schema:
        raise ValueError(f"unsupported joined dataset {directory} (schema {metadata.get('schema')!r})")

    with np.load(directory / "data.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}

    sizes = arrays["record_bytes"]
    keep = np.ones(len(sizes), dtype = bool) if max_bytes is None else sizes <= max_bytes
    if not keep.any():
        raise RuntimeError(f"axiom dataset contains no training examples: {directory}")

    examples = list(stored_examples(arrays, stored_graphs(arrays, keep)))
    failures = [
        {"problem": problem, "outcome": outcome}
        for problem, outcome in json.loads(arrays["failures"].tobytes()).items()
    ]

    return examples, failures, {
        "schema": axiom_dataset_schema,
        "collection": metadata["collection"],
        "shards": [source["path"] for source in metadata["inputs"]],
    }
