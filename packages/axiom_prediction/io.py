from collections.abc import Iterable, Iterator, Mapping, Sequence
import csv
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Any

def read_csv(path: str | Path, max_bytes: int | None = None) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    with Path(path).open("r", encoding = "utf-8", newline = "") as file:
        # Each record is one line, so an unterminated last line is a row still being appended.
        lines = (
            line
            for line in file
            if line.endswith("\n") and (max_bytes is None or len(line) <= max_bytes)
        )
        for row in csv.DictReader(lines, strict = True):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"corrupt dataset CSV row in {path}")

            yield {key: json.loads(value) for key, value in row.items() if value != ""}

def tmp_path(output: Path) -> Path:
    descriptor, path = tempfile.mkstemp(
        dir = output.parent,
        prefix = f".{output.name}.",
        suffix = ".tmp",
    )
    os.close(descriptor)
    return Path(path)

def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]):
    output = Path(path)
    tmp = tmp_path(output)
    try:
        with tmp.open("w", encoding = "utf-8", newline = "") as file:
            writer = csv.DictWriter(file, fieldnames = fields, lineterminator = "\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys = True) for key, value in row.items()})

        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok = True)

def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    default: Any = None,
):
    output = Path(path)
    tmp = tmp_path(output)
    try:
        tmp.write_text(
            json.dumps(payload, indent = 2, sort_keys = True, default = default),
            encoding = "utf-8",
        )

        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok = True)
