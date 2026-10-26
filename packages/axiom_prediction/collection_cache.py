import csv
import json
import os

def append_cache(path, payload, fields):
    with path.open("a", encoding = "utf-8", newline = "") as stream:
        writer = csv.DictWriter(stream, fieldnames = fields, lineterminator = "\n")
        writer.writerow({key: json.dumps(value, sort_keys = True) for key, value in payload.items()})
        stream.flush()
        os.fsync(stream.fileno())

def repair_cache(path):
    # JSON-encoded cells keep every CSV record on one physical line.
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end == 0:
            raise ValueError(f"empty collection cache: {path}")

        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return

        while end > 0:
            start = max(0, end - 8192)
            stream.seek(start)
            chunk = stream.read(end - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return

            end = start

        raise ValueError(f"collection cache has no complete header: {path}")
