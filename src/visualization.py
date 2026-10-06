"""Matplotlib style and figure helpers (one style source for all 17 charts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Single source of truth for report-grade figure style.
FIGURE_WIDTH_INCHES: float = 10.0
FIGURE_HEIGHT_INCHES: float = 5.5
FIGURE_DPI: int = 120
REPORT_FIGURES_DIRNAME: str = Path("reports") / "figures"


def apply_report_style() -> None:
    """Apply the project-wide plot style (idempotent, no figures created)."""
    plt.rcParams.update(
        {
            "figure.figsize": (FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
            "figure.dpi": FIGURE_DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def save_figure(fig: plt.Figure, filename: str, root: Path | str | None = None) -> Path:
    """Save a figure under ``reports/figures`` and return its path."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    out_path = base / REPORT_FIGURES_DIRNAME / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return out_path
