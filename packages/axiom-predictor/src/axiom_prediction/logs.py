from __future__ import annotations

from datetime import datetime
import sys

def log(message: str):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        "\n".join(f"{stamp} {line}" for line in str(message).split("\n")),
        file = sys.stderr,
        flush = True,
    )

__all__ = ["log"]
