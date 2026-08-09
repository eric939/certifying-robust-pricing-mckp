"""Publication plotting standards for robust_mckp figures.

The helpers in this module are intentionally Matplotlib-only and headless-safe.
Optional style packages may be present in some environments, but generated
manuscript figures must not depend on them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cycler import cycler


INK = "#1A1A1A"
GRID = "#E6E6E6"

# Okabe-Ito plus neutral grays.  The first six colors remain distinguishable in
# common colorblindness simulators and in grayscale when paired with markers.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": INK,
    "gray": "#787878",
    "light_gray": "#D9D9D9",
    "pale_blue": "#DCE8F2",
    "pale_orange": "#F5E3BF",
}

_SOLVER_DISPLAY = {
    "hullround": "HullRound",
    "global_bnb_baseline": r"$\theta$-B&B base",
    "global_bnb_cached": r"$\theta$-B&B cached",
    "global_bnb_cached_cutoff": r"$\theta$-B&B cutoff",
    "global_bnb_cached_cutoff_ordered": r"$\theta$-B&B opt.",
    "exact_enum_current": r"$\theta$ enumeration",
    "exact_sweep_new": r"$\theta$ sweep",
    "scipy_highs": "HiGHS",
    "highs": "HiGHS",
    "scip": "SCIP",
    "gurobi": "Gurobi",
    "cplex": "CPLEX",
}

_SOLVER_MARKER = {
    "hullround": "o",
    "global_bnb_baseline": "s",
    "global_bnb_cached_cutoff_ordered": "^",
    "exact_enum_current": "s",
    "exact_sweep_new": "o",
    "scipy_highs": "x",
    "highs": "x",
    "scip": "D",
    "gurobi": "P",
    "cplex": "v",
}

_SOLVER_LINESTYLE = {
    "hullround": "-",
    "global_bnb_baseline": "--",
    "global_bnb_cached_cutoff_ordered": "-",
    "exact_enum_current": "--",
    "exact_sweep_new": "-",
    "scipy_highs": "-.",
    "highs": "-.",
    "scip": ":",
    "gurobi": (0, (3, 1, 1, 1)),
    "cplex": (0, (1, 1)),
}


def safe_color_cycle() -> list[str]:
    """Return a colorblind-safe categorical color cycle."""

    return [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["sky"],
        COLORS["gray"],
        COLORS["black"],
    ]


def figure_size(kind: str = "double") -> tuple[float, float]:
    """Return manuscript-oriented figure sizes in inches."""

    presets = {
        "single": (3.45, 2.35),
        "single_tall": (3.45, 3.1),
        "double": (7.25, 4.6),
        "double_short": (7.25, 2.75),
        "double_tall": (7.25, 5.55),
        "appendix": (7.4, 5.8),
        "wide": (7.4, 3.0),
    }
    if kind not in presets:
        raise ValueError(f"unknown figure size preset: {kind}")
    return presets[kind]


def apply_top_journal_style() -> None:
    """Apply deterministic, headless-safe rcParams for manuscript figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
            "legend.handlelength": 2.2,
            "legend.borderaxespad": 0.2,
            "legend.columnspacing": 0.9,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.prop_cycle": cycler(color=safe_color_cycle()),
        }
    )


def solver_display_name(name: str) -> str:
    return _SOLVER_DISPLAY.get(name, name.replace("_", " "))


def solver_marker(name: str) -> str:
    return _SOLVER_MARKER.get(name, "o")


def solver_linestyle(name: str) -> Any:
    return _SOLVER_LINESTYLE.get(name, "-")


def format_log_axis(ax: Axes, *, axis: str = "y") -> None:
    """Apply readable grid styling to a log axis."""

    if axis in {"x", "both"}:
        ax.set_xscale("log")
        ax.grid(True, which="major", axis="x", color=GRID, linewidth=0.6)
        ax.grid(True, which="minor", axis="x", color="#F0F0F0", linewidth=0.4)
    if axis in {"y", "both"}:
        ax.set_yscale("log")
        ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.6)
        ax.grid(True, which="minor", axis="y", color="#F0F0F0", linewidth=0.4)


def annotate_time_limit(ax: Axes, time_limit: float, *, label: str | None = None) -> None:
    """Draw a horizontal time-limit reference line."""

    ax.axhline(time_limit, color="0.25", linestyle=":", linewidth=0.9)
    ax.annotate(
        label or f"{time_limit:g}s limit",
        xy=(0.99, time_limit),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.25",
    )


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def write_figure_metadata(path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write JSON metadata next to a figure PDF."""

    figure_path = Path(path)
    meta_path = figure_path.with_suffix(".metadata.json")
    payload: dict[str, Any] = {
        "figure": str(figure_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
    }
    if metadata:
        payload.update(dict(metadata))
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def save_figure(
    fig: Figure,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    png_preview: bool = True,
    close: bool = True,
) -> None:
    """Save a vector PDF figure and metadata, with optional PNG preview."""

    figure_path = Path(path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    if png_preview:
        fig.savefig(figure_path.with_suffix(".png"), bbox_inches="tight", dpi=400)
    write_figure_metadata(figure_path, metadata)
    if close:
        plt.close(fig)
