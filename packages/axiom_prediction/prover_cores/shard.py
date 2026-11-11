# Turns collect_cores.py rows into a dataset shard (problems.csv) that `join` reads.
import argparse
import json
from pathlib import Path
import sys

checkout = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(checkout / "packages"), str(checkout / "src")]

from axiom_prediction.dataset import axiom_dataset_shard_schema, dataset_fields
from axiom_prediction.io import write_csv
from axiom_prediction.prover_cores.provers import Prover, modules

def failure(row, outcome):
    return {"schema": axiom_dataset_shard_schema, "problem_path": row["problem_path"], "outcome": outcome, "parseable": "error" not in row}

# Rows collected before collect_cores.py had --max-example-mb can still hold larger examples.
def shard_rows(collection, inputs, max_example_mb):
    yield {"schema": axiom_dataset_shard_schema, "record": "metadata", "collection": collection}
    for path in inputs:
        for line in path.open():
            row = json.loads(line)
            if "example" not in row:
                yield failure(row, row.get("error") or "Timeout")
            elif len(line) > max_example_mb * 1e6:
                yield failure(row, "ExampleTooLarge")
            else:
                yield row["example"]

def main():
    parser = argparse.ArgumentParser(description = "write a dataset shard from collected prover cores")
    parser.add_argument("--prover", type = Prover, choices = list(Prover), required = True)
    parser.add_argument("--output", type = Path, required = True, help = "shard directory")
    parser.add_argument("--max-example-mb", type = float, default = 10)
    parser.add_argument("inputs", type = Path, nargs = "+")
    args = parser.parse_args()

    backend = args.prover.lower()
    collection = {"prover_backend": backend, "label_semantics": modules[args.prover].label_semantics, "seed": 0}
    args.output.mkdir(parents = True, exist_ok = True)
    write_csv(args.output / "problems.csv", shard_rows(collection, args.inputs, args.max_example_mb), dataset_fields)

if __name__ == "__main__":
    main()
