# Locked output journals and atomic evaluation checkpoints.

import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path

from .io import write_json_atomic

def resume_identity(identity):
    resume_options = {
        "output",
        "num_workers",
        "device",
        "multiprocess",

        "run_name",
        "wandb",
        "no_wandb",
    }

    if not isinstance(identity, dict):
        return identity

    return {key: value for key, value in identity.items() if key not in resume_options}

def csv_record(header, values):
    text_fields = {"event", "problem", "outcome", "tptp_status", "mode", "policy"}

    if len(values) != len(header):
        raise ValueError("corrupt CSV output record")

    row = dict(zip(header, values))
    for key, value in row.items():
        if key in text_fields or value == "":
            continue

        try:
            row[key] = json.loads(value)
        except ValueError:
            pass

    return row

def csv_records(content, fields):
    lines = list(io.StringIO(content, newline = ""))
    expected = ",".join(fields) + "\n"
    if not content or ("\n" not in content and expected.startswith(content)):
        return [], 0

    reader = csv.reader(lines, strict = True)
    if next(reader) != fields:
        raise ValueError("output CSV header does not match this command")

    records = []
    boundary = reader.line_num
    while True:
        try:
            values = next(reader)
        except StopIteration:
            break
        except csv.Error:
            if reader.line_num < len(lines):
                raise ValueError("corrupt CSV output")

            break

        if not lines[reader.line_num - 1].endswith("\n"):
            break

        records.append(csv_record(fields, values))
        boundary = reader.line_num

    return records, len("".join(lines[:boundary]).encode("utf-8"))

def json_records(content):
    records = []
    boundary = 0
    for line in io.StringIO(content, newline = ""):
        if not line.endswith("\n"):
            break

        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("output JSONL records must be objects")

        records.append(row)
        boundary += len(line.encode("utf-8"))

    return records, boundary

class OutputJournal:
    def __init__(self, path, command, use_csv, fields, identity):
        self.path = Path(path)
        self.command = command
        self.use_csv = use_csv
        self.fields = fields
        self.records = []
        self.keys = set()
        self.metrics = None

        self.directory = Path(str(path) + ".resume")
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self.stream = self.path.open(
            "a+",
            encoding = "utf-8",
            errors = "surrogateescape",
            newline = "",
        )

        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.restore(identity)
            self.stream.seek(0, 2)
            if self.use_csv:
                self.writer = csv.DictWriter(
                    self.stream,
                    fieldnames = fields,
                    lineterminator = "\n",
                )

                if self.stream.tell() == 0:
                    self.writer.writeheader()
                    self.stream.flush()
        except BaseException:
            self.stream.close()
            raise

    def restore(self, identity):
        self.stream.seek(0)
        content = self.stream.read()
        self.check_format(content)
        metadata = self.directory / "config.json"
        expected = resume_identity(json.loads(json.dumps(identity, default = str)))
        if (
            metadata.exists()
            and resume_identity(json.loads(metadata.read_text())) != expected
        ):
            raise ValueError(
                "output belongs to a different command/configuration; use a new --output"
            )

        self.records, boundary = (
            csv_records(content, self.fields) if self.use_csv else json_records(content)
        )

        self.restore_problem_paths()
        self.keys = {self.key(row) for row in self.records}
        self.directory.mkdir(parents = True, exist_ok = True)
        write_json_atomic(metadata, expected)
        # A torn final record may include an incomplete UTF-8 character.
        self.stream.truncate(boundary)
        self.stream.flush()

    def restore_problem_paths(self):
        for index, row in enumerate(self.records):
            saved = self.load("output-path", index)
            if saved and row.get("problem") in (
                Path(saved["problem"]).name, Path(saved["problem"]).stem
            ):
                row.update(saved)

    def check_format(self, content):
        if not content.strip():
            return

        first = content.lstrip()
        if self.use_csv and first.startswith("{"):
            raise ValueError(
                "existing output is JSONL, but --csv requests CSV; use the matching format or a new --output"
            )

        if not self.use_csv and not first.startswith("{"):
            raise ValueError(
                "existing output is not JSONL (it may be CSV); use --csv for CSV or a new --output"
            )

    @staticmethod
    def key(row):
        return row.get("event") or "problem", row.get("problem") or ""

    def complete(self):
        path = self.directory / "completion.json"
        if not path.exists():
            return False

        completion = json.loads(path.read_text())
        return completion["output_bytes"] == self.path.stat().st_size

    def exit_code(self):
        if self.complete():
            return json.loads((self.directory / "completion.json").read_text())["exit_code"]

        return (
            2
            if self.command == "run" and any(row.get("error") for row in self.records)
            else 0
        )

    def finish(self, exit_code):
        self.stream.flush()
        write_json_atomic(
            self.directory / "completion.json",
            {
                "output_bytes": self.path.stat().st_size,
                "exit_code": exit_code,
                "metrics": self.metrics,
            },
            default = str,
        )

    def write(self, row):
        from .output import csv_value, display_record

        key = self.key(row)
        if key in self.keys:
            return

        if row.get("problem"):
            metadata = {"problem": str(row["problem"])}
            if "outcome" in row:
                metadata["outcome"] = row["outcome"]

            self.save("output-path", len(self.records), metadata)

        record = display_record(row)
        if self.use_csv:
            self.writer.writerow({key: csv_value(value) for key, value in record.items()})
        else:
            self.stream.write(json.dumps(record, sort_keys = True, default = str) + "\n")

        self.stream.flush()
        self.keys.add(key)
        self.records.append(dict(row))

    def cache_path(self, stage, problem):
        key = hashlib.sha256(str(problem).encode()).hexdigest()
        return self.directory / f"{stage}-{key}.json"

    def load(self, stage, problem):
        path = self.cache_path(stage, problem)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, stage, problem, record):
        write_json_atomic(self.cache_path(stage, problem), record)

    def close(self):
        self.stream.close()
