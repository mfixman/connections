from collections.abc import Iterable, Iterator, Mapping
import json
import os
from pathlib import Path
from typing import Any
import uuid

def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding = "utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError(f"JSONL row in {path} must be an object")

            yield dict(row)

def tmp_path(output: Path) -> Path:
    # Concurrent writers need separate temporary files.
    return output.with_suffix(f"{output.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")

def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]):
    output = Path(path)
    tmp = tmp_path(output)
    try:
        with tmp.open("w", encoding = "utf-8") as file:
            for row in rows:
                file.write(json.dumps(dict(row), sort_keys = True))
                file.write("\n")

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
