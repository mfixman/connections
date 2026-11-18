# What collect results and joined datasets share: schema names, collection settings and the label policy.

from collections.abc import Mapping
from contextlib import contextmanager
import gc
import json
from pathlib import Path

from .choices import ProverPolicy
from .logs import log

axiom_dataset_schema = "learncop.axiom_prediction.dataset.v2"
# Rows of the CSV shards that preceded problems.csv; prover_cores still writes them and join reads them.
axiom_dataset_shard_schema = "learncop.axiom_prediction.dataset-shard.v2"
dataset_fields = (
    "schema record collection summary problem_path graph axiom_clause_ids "
    "conjecture_clause_ids labels axiom_clause_texts problem outcome parseable error"
).split()

# Training reads joined datasets only; `join` turns collect results into one.
def load_axiom_dataset(path, max_example_mb = None):
    from .joined import load_joined

    root = Path(path)
    max_bytes = None if max_example_mb is None else int(max_example_mb * 1e6)
    return load_joined(root.parent if root.suffix == ".npz" else root, max_bytes)

@contextmanager
def paused_gc():
    # Scanning the new objects takes over half of a load; freezing them spares training too.
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.freeze()
        if enabled:
            gc.enable()

# The settings that two results must share to count as one collection.
def collection_settings(collection):
    settings = {key: value for key, value in collection.items() if key != "tptp_root"}
    if settings.get("sat_policy") == "satcop" and settings.get("prover_backend") != "rust-satcop":
        settings["sat_policy"] = "satcopcon"

    if settings.get("seed") == 0:
        settings.pop("seed")

    return settings

def merged_collection(collections):
    first = next(filter(None, collections.values()), {})
    settings = {
        json.dumps(collection_settings(collection), sort_keys = True)
        for collection in collections.values()
    }

    if len(settings) <= 1:
        return first

    shared = {"sat_policy", "prover_backend", "label_semantics"}
    kept = {key: value for key, value in collection_settings(first).items() if key in shared}
    return {**kept, "source_shards": collections}

# The policy that labelled a dataset or checkpoint; before native SATCoP, `satcop` meant the Python SATCoPCon.
def label_policy(requested, metadata = None):
    collection = (metadata or {}).get("collection", {})
    recorded = collection.get("sat_policy") if isinstance(collection, Mapping) else None
    if recorded is not None:
        recorded = ProverPolicy(recorded)
        backend = collection.get("prover_backend") or collection.get("collection", {}).get("prover_backend")
        if recorded == ProverPolicy.SatCoP and backend != "rust-satcop":
            recorded = ProverPolicy.SatCoPCon

    if requested is not None and recorded is not None and ProverPolicy(requested) != recorded:
        log(f"warning: labels were collected with {recorded}, but --policy requests {requested}; running with it")

    return ProverPolicy(requested or recorded or ProverPolicy.SatResetCoP)

def read_object(path):
    value = json.loads(Path(path).read_text(encoding = "utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON record must be an object: {path}")

    return dict(value)
