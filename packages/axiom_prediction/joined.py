# Joins collect results and joined datasets into one deduplicated file of tight numerical arrays.

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

joined_schema = "learncop.axiom_prediction.joined.v2"

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

id_tables = ["axiom_ids", "conjecture_ids"]

@dataclass(slots = True)
class CompactGraph:
    name: str
    digest: bytes
    tables: dict[str, np.ndarray]
    texts: list[str]
    cores: dict[tuple[int, ...], set[tuple]]
    record_bytes: int

@dataclass(frozen = True, slots = True)
class JoinInput:
    inputs: list[dict]
    graphs: list[CompactGraph]
    outcomes: set[tuple]

def input_files(path):
    path = Path(path)
    if path.suffix == ".npz":
        return [path.parent.resolve()]

    if (path / "data.npz").exists():
        return [path.resolve()]

    if path.is_file():
        found = [path]
    else:
        # Old layouts publish each cache as SHARD.csv, a copy that may have been cut to a stub.
        caches = sorted(path.glob(".cache/*/cache.csv"))
        copies = {cache.parent.name for cache in caches}
        shards = [file for file in sorted(path.glob("*.csv")) if file.stem not in copies]
        found = caches + shards

    if not found:
        raise FileNotFoundError(f"no collect results in {path}")

    if any(file.suffix != ".csv" for file in found):
        raise ValueError(f"join reads CSV collect results and joined datasets, not {path}")

    return [file.resolve() for file in found]

def source_of(collection):
    return (
        collection.get("sat_policy", ""),
        collection.get("prover_backend") or "",
        collection.get("seed") or 0,
    )

def one_way_complements(rows, name):
    # Edges come in mirrored pairs; storing one side halves the largest table.
    scale = int(rows.max(initial = 0)) + 1
    forward = rows[:, 0].astype(np.int64) * scale + rows[:, 1]
    backward = rows[:, 1].astype(np.int64) * scale + rows[:, 0]
    if (rows[:, 0] == rows[:, 1]).any() or not np.array_equal(np.sort(forward), np.sort(backward)):
        raise ValueError(f"complement edges of {name} are not mirrored pairs")

    return rows[rows[:, 0] < rows[:, 1]]

def compact_row(name, row, record_bytes, source):
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
    texts = row.get("axiom_clause_texts", [])
    return CompactGraph(name, digest.digest(), tables, texts, {core: {source}}, record_bytes)

def outcome_row(name, row, source):
    outcome = str(row.get("outcome", "proved" if "graph" in row else "failed"))
    parseable = row.get("parseable", not outcome.startswith("TPTPParseError:"))
    return name, source, outcome, bool(parseable), bool(row.get("error", False))

def read_input(path):
    known_schemas = {problem_schema, axiom_example_schema, axiom_dataset_schema, axiom_dataset_shard_schema}

    graphs, outcomes = [], set()
    collection = None
    for record_bytes, row in sized_csv_rows(path):
        if row.get("schema") not in known_schemas:
            raise ValueError(f"unsupported schema {row.get('schema')!r} in {path}")

        collection = collection or row.get("collection")
        if row.get("record") == "metadata":
            continue

        source = source_of(collection or {})
        name = problem_name(row.get("problem_name") or row.get("problem_path") or row["problem"])
        outcomes.add(outcome_row(name, row, source))
        if "graph" in row:
            graphs.append(compact_row(name, row, record_bytes, source))

    if not outcomes:
        raise ValueError(f"{path} has no results")

    proved = sum(row[2] == "proved" for row in outcomes)
    summary = {
        "path": str(path),
        "collection": collection or {},
        "proved": proved,
        "failed": len(outcomes) - proved,
    }

    return JoinInput([summary], graphs, outcomes)

def read_metadata(directory):
    metadata = read_object(directory / "metadata.json")
    if metadata.get("schema") != joined_schema:
        raise ValueError(f"unsupported joined dataset {directory} (schema {metadata.get('schema')!r})")

    return metadata

def kept_rows(arrays, name, keep = None):
    counts = arrays[f"{name}.counts"]
    if keep is None:
        return np.split(arrays[name], np.cumsum(counts)[:-1])

    return np.split(arrays[name][np.repeat(keep, counts)], np.cumsum(counts[keep])[:-1])

def stored_texts(arrays):
    texts = json.loads(arrays["clause_texts"].tobytes())
    stops = np.cumsum(arrays["clause_texts.counts"]).tolist()
    return [texts[start : stop] for start, stop in zip([0, *stops], stops)]

def read_joined(directory):
    metadata = read_metadata(directory)
    with np.load(directory / "data.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}

    log(f"read joined dataset {directory}")
    tables = {name: kept_rows(arrays, name) for name in (*table_widths, *id_tables)}
    texts = stored_texts(arrays)
    graphs = [
        CompactGraph(
            name,
            arrays["graph_digests"][index].tobytes(),
            {table: rows[index] for table, rows in tables.items()},
            texts[index],
            {},
            size,
        )
        for index, (name, size) in enumerate(zip(arrays["problem_names"].tolist(), arrays["record_bytes"].tolist()))
    ]

    sources = list(map(tuple, metadata["sources"]))
    positions = kept_rows(arrays, "core_positions")
    finders = kept_rows(arrays, "core_sources")
    for index, core, found_by in zip(arrays["example_graphs"].tolist(), positions, finders, strict = True):
        graphs[index].cores[tuple(core.tolist())] = {sources[source] for source in found_by.tolist()}

    outcomes = {
        (name, sources[source], outcome, parseable, error)
        for name, source, outcome, parseable, error in json.loads(arrays["outcomes"].tobytes())
    }

    return JoinInput(metadata["inputs"], graphs, outcomes)

def read_inputs(files, num_workers):
    yield from map(read_joined, filter(Path.is_dir, files))

    collected = [file for file in files if not file.is_dir()]
    if not collected:
        return

    workers = determine_worker_count(len(collected), num_workers)
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer = gc.disable) as pool:
        results = pool.imap_unordered(read_input, collected)
        yield from track_progress(results, "reading collect results", len(collected))

def merge_graph(graphs, graph):
    kept = graphs.setdefault((graph.name, graph.digest), graph)
    kept.record_bytes = min(kept.record_bytes, graph.record_bytes)
    if graph.texts and (not kept.texts or graph.texts < kept.texts):
        kept.texts = graph.texts

    for core, sources in graph.cores.items():
        kept.cores.setdefault(core, set()).update(sources)

def distinct_inputs(summaries):
    # One file can arrive twice, directly and through a joined dataset; keep its fullest state.
    ordered = sorted(summaries, key = lambda summary: (summary["path"], summary["proved"] + summary["failed"]))
    return list({summary["path"]: summary for summary in ordered}.values())

def narrow(array):
    if array.size == 0 or array.min() < 0:
        return array

    return array.astype(np.min_scalar_type(int(array.max())))

def add_table(arrays, name, rows):
    arrays[name] = narrow(np.concatenate(rows))
    arrays[f"{name}.counts"] = np.asarray(list(map(len, rows)), dtype = np.int64)

def json_array(value):
    return np.frombuffer(json.dumps(value, sort_keys = True).encode(), dtype = np.uint8)

def failure_outcomes(outcomes):
    failures = {}
    for name, _, outcome, _, _ in sorted(outcomes):
        if outcome != "proved":
            failures.setdefault(name, outcome)

    return failures

def stored_arrays(graphs, outcomes, sources):
    keys = sorted(graphs)
    positions = {source: index for index, source in enumerate(sources)}
    examples = [
        (index, core, graphs[key].cores[core])
        for index, key in enumerate(keys)
        for core in sorted(graphs[key].cores)
    ]

    outcome_rows = [
        [name, positions[source], outcome, parseable, error]
        for name, source, outcome, parseable, error in sorted(outcomes)
    ]

    arrays = {
        "problem_names": np.asarray([name for name, _ in keys]),
        "graph_digests": np.frombuffer(b"".join(digest for _, digest in keys), dtype = np.uint8).reshape(-1, 16),
        "record_bytes": np.asarray([graphs[key].record_bytes for key in keys], dtype = np.int64),
        "example_graphs": np.asarray([example[0] for example in examples], dtype = np.int64),

        "failures": json_array(failure_outcomes(outcomes)),
        "outcomes": json_array(outcome_rows),
        "clause_texts": json_array([text for key in keys for text in graphs[key].texts]),
        "clause_texts.counts": np.asarray([len(graphs[key].texts) for key in keys], dtype = np.int64),
    }

    # Popping each table lets its per-graph arrays go once they are concatenated.
    for table in (*table_widths, *id_tables):
        add_table(arrays, table, [graphs[key].tables.pop(table) for key in keys])

    finders = [sorted(positions[source] for source in example[2]) for example in examples]
    add_table(arrays, "core_positions", [np.asarray(example[1], dtype = np.int32) for example in examples])
    add_table(arrays, "core_sources", [np.asarray(found, dtype = np.int32) for found in finders])
    return arrays

def write_arrays(path, arrays):
    tmp = tmp_path(path)
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok = True)

def joined_description(summaries, sources, counts):
    collections = {summary["path"]: summary["collection"] for summary in summaries}
    parts = {part for path in collections for part in re.findall(r"part-(\d+)", path)}
    return {
        "schema": joined_schema,
        "collection": merged_collection(collections),
        "provers": sorted({source[0] for source in sources}),
        "seeds": sorted({source[2] for source in sources}),
        "parts": sorted(map(int, parts)),
        "sources": sources,
        **counts,
        "inputs": summaries,
    }

@paused_gc()
def join_datasets(inputs, output_dir, num_workers = None):
    output = Path(output_dir)
    data, description = output / "data.npz", output / "metadata.json"
    if data.exists() or description.exists():
        raise FileExistsError(f"refusing to replace the joined dataset in {output}")

    files = list(dict.fromkeys(file for path in inputs for file in input_files(path)))
    graphs, outcomes, summaries = {}, set(), []
    for result in read_inputs(files, num_workers):
        summaries += result.inputs
        outcomes |= result.outcomes
        for graph in result.graphs:
            merge_graph(graphs, graph)

    if not graphs:
        raise RuntimeError(f"no proved problems in the {len(files)} inputs")

    summaries = distinct_inputs(summaries)
    sources = sorted({row[1] for row in outcomes})
    proved = {name for name, _ in graphs}
    counts = {
        "problems": len(proved),
        "graphs": len(graphs),
        "examples": sum(len(graph.cores) for graph in graphs.values()),
        "failures": len(failure_outcomes(outcomes).keys() - proved),
    }

    log(f"joined {len(summaries)} inputs: {counts}; writing {data}")
    output.mkdir(parents = True, exist_ok = True)
    write_arrays(data, stored_arrays(graphs, outcomes, sources))
    write_json_atomic(description, joined_description(summaries, sources, counts))
    return {"inputs": len(summaries), **counts, "data_bytes": data.stat().st_size, "dataset": str(output)}

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
    texts = stored_texts(arrays)
    cores = kept_rows(arrays, "core_positions")
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
            axiom_clause_texts = texts[index],
        )

@paused_gc()
def load_joined(directory, max_bytes = None):
    provenance = {"outcomes", "core_sources", "core_sources.counts", "graph_digests"}

    metadata = read_metadata(directory)
    with np.load(directory / "data.npz") as stored:
        arrays = {key: stored[key] for key in stored.files if key not in provenance}

    sizes = arrays["record_bytes"]
    keep = np.ones(len(sizes), dtype = bool) if max_bytes is None else sizes <= max_bytes
    if not keep.any():
        raise RuntimeError(f"axiom dataset contains no training examples: {directory}")

    examples = list(stored_examples(arrays, stored_graphs(arrays, keep)))

    # A problem whose graphs are all too large counts as unproved, as in the CSV loader.
    proved = set(arrays["problem_names"][keep].tolist())
    failures = [
        {"problem": problem, "outcome": outcome}
        for problem, outcome in json.loads(arrays["failures"].tobytes()).items()
        if problem not in proved
    ]

    return examples, failures, {
        "schema": axiom_dataset_schema,
        "collection": metadata["collection"],
        "shards": [summary["path"] for summary in metadata["inputs"]],
    }
