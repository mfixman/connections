#!/usr/bin/env python3
"""Render Mangla-style violin plots for schedule scores."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DISPLAY_LABELS = {
    "Portfolio60Novel": "P2 60s",
    "Portfolio60Total": "P1 60s",
    "Portfolio120Novel": "P2 120s",
    "Portfolio120Total": "P1 120s",
}

ORDER = (
    "Portfolio60Novel",
    "Portfolio60Total",
    "Portfolio120Novel",
    "Portfolio120Total",
)

OBJECTIVE_COLORS = {
    "novel": "#c97b2c",
    "total": "#2a6f97",
}

@dataclass(frozen=True)
class ScheduleStats:
    schedule: str
    budget_label: str
    objective_label: str
    solved: int
    total: int
    mean_score: float

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot violin plots for schedule scores.",
    )
    parser.add_argument(
        "--input",
        default="results/schedule_violin_data.csv",
        help="Long-form CSV produced by schedule_violin_data.py.",
    )
    parser.add_argument(
        "--combined-output",
        default="results/schedule_violin_combined.png",
        help="Output path for the combined four-schedule figure.",
    )
    parser.add_argument(
        "--combined-pdf",
        default="results/schedule_violin_combined.pdf",
        help="Optional PDF output for the combined figure.",
    )
    parser.add_argument(
        "--budget-output",
        default="results/schedule_violin_by_budget.png",
        help="Output path for the split-by-budget figure.",
    )
    parser.add_argument(
        "--budget-pdf",
        default="results/schedule_violin_by_budget.pdf",
        help="Optional PDF output for the split-by-budget figure.",
    )
    return parser.parse_args()

def _objective_label(schedule: str) -> str:
    lower = schedule.lower()
    if "novel" in lower:
        return "novel"
    if "total" in lower:
        return "total"
    return "other"

def _budget_label(schedule: str) -> str:
    if "120" in schedule:
        return "120s"
    if "60" in schedule:
        return "60s"
    return "other"

def load_scores(
    csv_path: str,
) -> tuple[dict[str, list[float]], dict[str, ScheduleStats]]:
    scores: dict[str, list[float]] = defaultdict(list)
    solved_counts: dict[str, int] = defaultdict(int)

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"schedule", "score", "solved"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or ())
            raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            schedule = row["schedule"].strip()
            if not schedule:
                continue
            scores[schedule].append(float(row["score"]))
            if row["solved"].strip().lower() == "true":
                solved_counts[schedule] += 1

    stats: dict[str, ScheduleStats] = {}
    for schedule, values in scores.items():
        stats[schedule] = ScheduleStats(
            schedule=schedule,
            budget_label=_budget_label(schedule),
            objective_label=_objective_label(schedule),
            solved=solved_counts[schedule],
            total=len(values),
            mean_score=sum(values) / len(values),
        )
    return scores, stats

def _style_violin_parts(parts, colors: list[str]) -> None:
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.85)
        body.set_linewidth(0.8)

    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("#222222")
            artist.set_linewidth(0.8)

def _annotate_counts(ax, x_positions: list[int], names: list[str], stats: dict[str, ScheduleStats]) -> None:
    for xpos, name in zip(x_positions, names):
        entry = stats[name]
        ax.text(
            xpos,
            -0.08,
            f"{entry.solved}/{entry.total}",
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
            clip_on=False,
        )

def plot_combined(
    scores: dict[str, list[float]],
    stats: dict[str, ScheduleStats],
    output_png: str,
    output_pdf: str,
) -> None:
    names = [name for name in ORDER if name in scores]
    values = [scores[name] for name in names]
    colors = [OBJECTIVE_COLORS.get(stats[name].objective_label, "#888888") for name in names]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    parts = ax.violinplot(values, showmeans=True, showextrema=True, widths=0.9)
    _style_violin_parts(parts, colors)

    x_positions = list(range(1, len(names) + 1))
    ax.set_xticks(x_positions, [DISPLAY_LABELS.get(name, name) for name in names])
    ax.set_ylim(-0.12, 1.02)
    ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
    ax.text(
        len(names) + 0.4,
        0.505,
        "success threshold",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.set_ylabel("Mangla score")
    ax.set_xlabel("Schedule")
    ax.set_title("Score Distributions for Static Schedules")
    ax.text(
        0.01,
        0.98,
        "score = 0.5 + (T - t)/(2T) on success; otherwise (T - t)/(2T)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    _annotate_counts(ax, x_positions, names, stats)
    ax.text(
        0.01,
        -0.12,
        "Solved counts shown below each violin",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

def plot_by_budget(
    scores: dict[str, list[float]],
    stats: dict[str, ScheduleStats],
    output_png: str,
    output_pdf: str,
) -> None:
    grouped = {
        "60s": [name for name in ORDER if name in scores and stats[name].budget_label == "60s"],
        "120s": [name for name in ORDER if name in scores and stats[name].budget_label == "120s"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True, constrained_layout=True)
    for ax, budget in zip(axes, ("60s", "120s")):
        names = grouped[budget]
        values = [scores[name] for name in names]
        colors = [OBJECTIVE_COLORS.get(stats[name].objective_label, "#888888") for name in names]
        parts = ax.violinplot(values, showmeans=True, showextrema=True, widths=0.9)
        _style_violin_parts(parts, colors)

        x_positions = list(range(1, len(names) + 1))
        ax.set_xticks(x_positions, [DISPLAY_LABELS.get(name, name).split(" ", 1)[1] for name in names])
        ax.set_ylim(-0.12, 1.02)
        ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
        ax.set_xlabel(budget)
        ax.set_title(f"{budget} schedules")
        _annotate_counts(ax, x_positions, names, stats)

    axes[0].set_ylabel("Mangla score")
    fig.suptitle("Schedule Score Distributions by Budget", fontsize=13)
    axes[0].text(
        0.0,
        -0.14,
        "Solved counts shown below each violin",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    args = parse_args()
    scores, stats = load_scores(args.input)

    Path(args.combined_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.budget_output).parent.mkdir(parents=True, exist_ok=True)

    plot_combined(
        scores=scores,
        stats=stats,
        output_png=args.combined_output,
        output_pdf=args.combined_pdf,
    )
    plot_by_budget(
        scores=scores,
        stats=stats,
        output_png=args.budget_output,
        output_pdf=args.budget_pdf,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
