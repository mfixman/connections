from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

split_scheme = 3

@dataclass(slots = True, init = False)
class ProblemSplit:
    split: int
    parts: list[int]

    def __init__(self, split: int = 1, parts: list[int] | None = None):
        self.split = split
        self.parts = [0] if parts is None else parts

        self.validate()

    def validate(self):
        if self.split < 1:
            raise ValueError("--split must be at least 1")

        if not self.parts:
            raise ValueError("--parts needs at least one part")

        if any(p < 0 or p >= self.split for p in self.parts):
            raise ValueError(
                f"--parts must be between 0 and {self.split - 1} for --split {self.split}"
            )

        if len(set(self.parts)) != len(self.parts):
            raise ValueError("--parts must not repeat a part")

    @classmethod
    def everything(cls) -> "ProblemSplit":
        return cls()

    def is_everything(self) -> bool:
        return len(self.parts) == self.split

    def part(self, problem: str | Path) -> int:
        if self.split == 1:
            return 0

        digest = hashlib.blake2b(
            f"{self.split}\0{split_key(problem)}".encode("utf-8"),
            digest_size = 8,
            person = b"lcsplit",
        ).digest()

        return int.from_bytes(digest, "big") % self.split

    def contains(self, problem: str | Path) -> bool:
        return self.is_everything() or self.part(problem) in self.parts

    def select(self, problems: list[str] | tuple[str, ...]) -> list[str]:
        return [p for p in problems if self.contains(p)]

    def describe(self) -> str:
        if self.is_everything():
            return "every problem"

        return f"part{'s' if len(self.parts) > 1 else ''} {' '.join(map(str, self.parts))} of {self.split} (by problem prefix)"

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "parts": list(self.parts),
            "scheme": split_scheme,
        }

def split_key(problem: str | Path) -> str:
    _tptp_name = re.compile(r"[A-Z]{3}\d{3}")

    name = Path(problem).stem
    match = _tptp_name.match(name)
    return match.group() if match else name
