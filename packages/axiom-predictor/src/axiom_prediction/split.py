from __future__ import annotations

from .choices import SplitKey

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

SPLIT_KEYS = tuple(SplitKey)
SPLIT_SCHEME = 2
_TPTP_NAME = re.compile(r"([A-Z]{3})(\d{3})")

@dataclass(frozen = True, slots = True)
class ProblemSplit:
    split: int = 1
    parts: tuple[int, ...] = (0,)
    by: SplitKey | str = SplitKey.Filename

    def __post_init__(self):
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

        object.__setattr__(self, "by", SplitKey(self.by))

    @classmethod
    def everything(cls) -> "ProblemSplit":
        return cls()

    @property
    def is_everything(self) -> bool:
        return len(self.parts) == self.split

    def part(self, problem: str | Path) -> int:
        if self.split == 1:
            return 0

        digest = hashlib.blake2b(
            f"{self.split}\0{split_key(problem, by = self.by)}".encode("utf-8"),
            digest_size = 8,
            person = b"lcsplit",
        ).digest()

        return int.from_bytes(digest, "big") % self.split

    def contains(self, problem: str | Path) -> bool:
        return self.is_everything or self.part(problem) in self.parts

    def select(self, problems: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        return tuple(p for p in problems if self.contains(p))

    def describe(self) -> str:
        if self.is_everything:
            return "every problem"

        return f"part{'s' if len(self.parts) > 1 else ''} {' '.join(map(str, self.parts))} of {self.split} (by {self.by})"

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "parts": list(self.parts),
            "by": SplitKey(self.by).wire_value,
            "scheme": SPLIT_SCHEME,
        }

def split_key(problem: str | Path, *, by: str = SplitKey.Filename) -> str:
    by = SplitKey(by)
    name = Path(str(problem)).name
    if by == SplitKey.Filename:
        return name

    if name.endswith(".p"):
        name = name[:-2]

    if by == SplitKey.Problem:
        return name

    m = _TPTP_NAME.match(name)
    if m is None:
        return name

    return m.group(1) if by == SplitKey.Domain else m.group(1) + m.group(2)

__all__ = ["ProblemSplit", "SPLIT_KEYS", "SPLIT_SCHEME", "split_key"]
