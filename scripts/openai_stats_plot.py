#!/usr/bin/env python3
"""
Create a horizontal bar plot from a CSV with month labels and numeric values.

The script automatically detects:
- the month column (column containing 'month')
- the numeric value column (first numeric column)

Usage:
    python month_barplot.py path/to/file.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from genai_detection.config import CONFIG

MONTH_FORMAT = "%B %Y"


def detect_columns(df: pd.DataFrame):
    """Automatically detect month and value columns."""
    month_candidates = [c for c in df.columns if "month" in c.lower()]
    if not month_candidates:
        raise ValueError(f"No month column detected. Columns: {list(df.columns)}")

    month_col = month_candidates[0]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric column detected for values.")

    value_col = numeric_cols[0]
    return month_col, value_col


def load_data(csv_path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)

    month_col, value_col = detect_columns(df)

    df = df[[month_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="raise")
    df["_month_dt"] = pd.to_datetime(df[month_col], format=MONTH_FORMAT, errors="raise")
    df = df.sort_values("_month_dt")

    return df.reset_index(drop=True), month_col, value_col


def format_latex_thousands(value: float | int) -> str:
    """Format numbers like 100\\,000\\,000 for LaTeX/mathtext rendering."""
    return f"${value:,}$".replace(",", r"\,")


def add_value_labels(ax: plt.Axes, bars, values):
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin if xmax > xmin else 1
    pad = span * 0.02

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2
        width = bar.get_width()
        label = format_latex_thousands(value)

        if width > span * 0.12:
            x = width - pad
            ax.text(x, y, label, va="center", ha="right", color="black")
        else:
            ax.text(width + pad, y, label, va="center", ha="left", color="black")


def make_plot(df: pd.DataFrame, month_col: str, value_col: str, output: Path):
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(
        df[month_col],
        df[value_col],
        color="#d9d9d9",
        edgecolor="#333333",
        linewidth=1.2,
    )

    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#bfbfbf", alpha=0.6)

    max_val = float(df[value_col].max())
    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1)

    # Prevent scientific notation like 1e7 or 1e8
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)

    # Format x-axis ticks as 100\,000\,000
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_latex_thousands(x)))

    add_value_labels(ax, bars, df[value_col].tolist())

    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main(path2file):
    LOCAL_SAVE_PATH = (
        Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "openai_stats"
    )
    LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    csv_path = Path(__file__).resolve().parents[1] / path2file
    df, month_col, value_col = load_data(csv_path)

    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in value_col)
    savepath = LOCAL_SAVE_PATH / f"{safe_name}_barplot.pdf"

    make_plot(
        df=df,
        month_col=month_col,
        value_col=value_col,
        output=savepath,
    )

    print(f"Saved plot to {savepath}")


if __name__ == "__main__":
    main("scripts/costs_month_data.csv")
    main("scripts/n_requests_month_data.csv")
    main("scripts/tokens_month_data.csv")