"""Locked output journals and atomic evaluation checkpoints."""
import csv
import fcntl
import hashlib
import json
from pathlib import Path

from .io import write_json_atomic


class OutputJournal:
    def __init__(self, path, command, use_csv, fields, identity):
        self.path = Path(path)
        self.command = command
        self.use_csv = use_csv
        self.fields = fields
        self.records = []
        self.directory = Path(str(path) + '.resume')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+', encoding='utf-8', newline='')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream.seek(0)
            content = self.stream.read()
            if content.strip():
                first = content.lstrip()
                if use_csv and first.startswith("{"):
                    raise ValueError("existing output is JSONL, but --csv requests CSV; use the matching format or a new --output")
                if not use_csv and not first.startswith("{"):
                    raise ValueError("existing output is not JSONL (it may be CSV); use --csv for CSV or a new --output")
            self.directory.mkdir(parents=True, exist_ok=True)
            metadata = self.directory / 'config.json'
            expected = json.loads(json.dumps(identity, default=str))
            if metadata.exists() and json.loads(metadata.read_text()) != expected:
                raise ValueError('output belongs to a different command/configuration; use a new --output')
            self._read(content)
            write_json_atomic(metadata, expected)
            self.stream.seek(0, 2)
            if self.use_csv:
                self.writer = csv.DictWriter(self.stream, fieldnames=fields, lineterminator='\n')
                if self.stream.tell() == 0:
                    self.writer.writeheader()
                    self.stream.flush()
        except BaseException:
            self.stream.close()
            raise

    def _read(self, content):
        # Only complete newline-terminated logical records are committed.
        lines = content.splitlines(keepends=True)
        boundary = 0
        if self.use_csv:
            expected_header = ','.join(self.fields) + '\n'
            if content and '\n' not in content and expected_header.startswith(content):
                self.stream.truncate(0)
                return
            reader = csv.reader(lines, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                return
            if header != self.fields:
                raise ValueError('output CSV header does not match this command')
            boundary = reader.line_num
            while True:
                try:
                    values = next(reader)
                except StopIteration:
                    break
                except csv.Error:
                    if reader.line_num < len(lines):
                        raise ValueError('corrupt CSV output')
                    break
                if not lines[reader.line_num - 1].endswith('\n'):
                    break
                if len(values) != len(header):
                    raise ValueError('corrupt CSV output record')
                row = dict(zip(header, values))
                for key in row:
                    if key not in {'event', 'problem', 'outcome', 'mode', 'policy', 'error', 'guidance_fallback'}:
                        try:
                            row[key] = json.loads(row[key])
                        except (ValueError, TypeError):
                            pass
                self.records.append(row)
                boundary = reader.line_num
        else:
            for i, line in enumerate(lines):
                if not line.endswith('\n'):
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('output JSONL records must be objects')
                self.records.append(row)
                boundary = i + 1
        if boundary < len(lines):
            # Truncate a torn final record, preserving all committed records.
            self.stream.truncate(len(''.join(lines[:boundary]).encode('utf-8')))
            self.stream.flush()
        self.keys = {self.key(row) for row in self.records}

    @staticmethod
    def key(row):
        return row.get('event') or 'problem', row.get('problem') or ''

    @property
    def complete(self):
        return any(row.get('event') == 'summary' for row in self.records)

    def write(self, row):
        key = self.key(row)
        if key in getattr(self, 'keys', set()):
            return
        if self.use_csv:
            self.writer.writerow({k: v if isinstance(v, str) else json.dumps(v, default=str)
                                  for k, v in row.items()})
        else:
            self.stream.write(json.dumps(row, sort_keys=True, default=str) + '\n')
        self.stream.flush()
        self.keys = getattr(self, 'keys', set()) | {key}

    def cache_path(self, stage, problem):
        key = hashlib.sha256(str(problem).encode()).hexdigest()
        return self.directory / f'{stage}-{key}.json'

    def load(self, stage, problem):
        path = self.cache_path(stage, problem)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, stage, problem, record):
        write_json_atomic(self.cache_path(stage, problem), record)

    def close(self):
        self.stream.close()
