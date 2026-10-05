from __future__ import annotations

import re

from pycop.leancop import leancop_agent
from connections.interaction.strategy import MatrixOptions, PolicyOptions, Strategy


class LeancopSettingsCodec:
    """Encode/decode leanCoP-compatible strategy tokens for pycop schedules."""

    @staticmethod
    def to_tokens(strategy: Strategy | None = None) -> list[str]:
        normalized = strategy if strategy is not None else LeancopSettingsCodec.from_tokens(None)
        matrix = normalized.matrix
        if normalized.policy.policy_class is not leancop_agent:
            raise TypeError("leanCoP settings codec requires the leanCoP agent factory")

        args = normalized.policy.args or {}
        tokens: list[str] = []

        if matrix.translation in {"def", "nodef"}:
            tokens.append(matrix.translation)
        if matrix.mark_conjecture:
            tokens.append("conj")
        if matrix.reorder > 0:
            tokens.append(f"reo({matrix.reorder})")

        if args.get("cut", False):
            tokens.append("cut")
        if args.get("scut", False):
            tokens.append("scut")
        if args.get("comp") is not None:
            tokens.append(f"comp({args['comp']})")

        return tokens

    @staticmethod
    def to_token_list(strategy: Strategy | None = None) -> str:
        return f"[{','.join(LeancopSettingsCodec.to_tokens(strategy))}]"

    @staticmethod
    def split_token_list(text: str) -> list[str]:
        """leanCoP's list syntax: "[cut,comp(7)]", brackets optional.

        Commas inside parentheses do not split, so "reo(2),comp(7)" and
        "[def,conj,cut]" both parse.
        """
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        tokens: list[str] = []
        depth = 0
        current = ""
        for ch in text:
            if ch == "," and depth == 0:
                if current.strip():
                    tokens.append(current.strip())
                current = ""
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            current += ch
        if current.strip():
            tokens.append(current.strip())
        return tokens

    @staticmethod
    def from_tokens(tokens: list[str] | None) -> Strategy:
        translation = "default"
        reorder = 0
        conjecture = False
        cut = False
        scut = False
        comp: int | None = None

        for token in tokens or []:
            if token == "nodef":
                translation = "nodef"
                continue
            if token == "def" and translation != "nodef":
                translation = "def"
                continue
            if token == "conj":
                conjecture = True
                continue

            match = re.fullmatch(r"reo[(=](\d+)\)?", token)
            if match:
                reorder = int(match.group(1))
                continue

            if token == "cut":
                cut = True
                continue
            if token == "scut":
                scut = True
                continue

            match = re.fullmatch(r"comp[(=](\d+)\)?", token)
            if match:
                comp = int(match.group(1))

        return Strategy(
            matrix=MatrixOptions(
                translation=translation,
                reorder=reorder,
                mark_conjecture=conjecture,
            ),
            policy=PolicyOptions(
                policy_class=leancop_agent,
                args = {
                    "cut": cut,
                    "scut": scut,
                    "comp": comp,
                    "start": "conjecture" if conjecture else "positive",
                    "factorization": "equal",
                },
            ),
        )


__all__ = ["LeancopSettingsCodec"]
