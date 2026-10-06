"""Shared data-loading and preparation helpers.

Primitives (paths, timestamps), sessionization, and duplicate policy.
Point-in-time joins live in ``src.analysis`` (they are analytical, not
plumbing) — see ``as_of_item_attributes``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility / project constants (named, no magic numbers in notebooks)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# Timestamps in the raw files are integer epoch *milliseconds* (Stage 1 fact:
# events span 2015-05-03 to 2015-09-18). Keep the unit explicit everywhere.
TIMESTAMP_UNIT_MS: str = "ms"

# Candidate session-gap thresholds to *compare* in the sessionization analysis
# (Q2). The default is a starting point for sensitivity analysis, not a claim.
SESSION_GAP_MINUTES_DEFAULT: int = 30
SESSION_GAP_MINUTES_CANDIDATES: tuple[int, ...] = (15, 30, 60)

# Structural facts from Stage 1 (used for validation, not analysis):
EXPECTED_EVENT_TYPES: tuple[str, ...] = ("view", "addtocart", "transaction")
TRANSACTION_EVENT: str = "transaction"

# Raw data layout (relative to repository root).
RAW_DATA_DIRNAME: str = Path("data") / "raw"
EVENTS_FILENAME: str = "events.csv"
CATEGORY_TREE_FILENAME: str = "category_tree.csv"
ITEM_PROPERTIES_FILENAMES: tuple[str, ...] = (
    "item_properties_part1.csv",
    "item_properties_part2.csv",
)


def repo_root(start: Path | str = __file__) -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(start).resolve().parent.parent


def raw_data_dir(root: Path | str | None = None) -> Path:
    """Return the ``data/raw`` directory for a given repo root."""
    base = Path(root) if root is not None else repo_root()
    return base / "data" / "raw"


def to_utc_datetime(
    epoch_millis: pd.Series | pd.DataFrame | int,
) -> pd.Series | pd.DataFrame | pd.Timestamp:
    """Convert integer epoch-milliseconds to UTC datetime(s).

    Pure helper only — no analysis, no imputation, no timezone guessing.
    """
    return pd.to_datetime(epoch_millis, unit=TIMESTAMP_UNIT_MS, utc=True)


def list_raw_files(root: Path | str | None = None) -> list[Path]:
    """List expected raw files that exist (light inventory, no parsing)."""
    data_dir = raw_data_dir(root)
    candidates = [EVENTS_FILENAME, CATEGORY_TREE_FILENAME, *ITEM_PROPERTIES_FILENAMES]
    return [data_dir / name for name in candidates if (data_dir / name).exists()]


# ---------------------------------------------------------------------------
# Sessionization + duplicate policy (analytical joins live in src.analysis)
# ---------------------------------------------------------------------------

def sessionize_events(
    events: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES_DEFAULT,
) -> pd.DataFrame:
    """Assign session ids per visitor after an inactivity gap.

    A new session starts at each visitor's first event and whenever the gap
    to the previous event by the same visitor exceeds ``gap_minutes``.
    Session ids are dense integers in (visitor, time) order. Input must
    contain ``visitorid`` and ``timestamp`` (epoch ms). The 30-minute default
    is the Stage 5 sensitivity choice (15/30/60 compared in §7), not a claim.
    """
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive")
    ordered = events.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    gaps = ordered.groupby("visitorid")["timestamp"].diff() / 60000
    new_session = gaps.isna() | (gaps > gap_minutes)
    out = ordered.copy()
    out["session_id"] = new_session.cumsum().astype("int64")
    return out


def summarize_duplicates(events: pd.DataFrame) -> dict:
    """Count fully-duplicated event rows (Stage 4 policy: report, never drop).

    Note: subset-column dedup inflates counts (1.3M on 3 cols vs 460
    full-row) — always deduplicate on the full row.
    """
    return {
        "rows": int(len(events)),
        "full_row_duplicates": int(events.duplicated().sum()),
        "full_row_duplicate_pct": round(float(events.duplicated().mean()) * 100, 4),
    }
