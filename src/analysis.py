"""Analytical helpers (reviewed).

Each function maps to one ranked question from Stage 2 (Q1-Q10).
Leakage-sensitive joins are point-in-time only; naive latest-value joins
are quantified (not used) via as_of_vs_naive_disagreement.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data_processing import sessionize_events


def funnel_rates(events: pd.DataFrame) -> dict:
    """Event-level funnel with conditional rates per 1000 views (Q1).

    Rates are associational (no causal reading): tx-per-cart is an upper
    bound because multi-item baskets and cross-session carts break exact
    view→cart→buy linkage at event grain.
    """
    counts = events["event"].value_counts()
    n_view = int(counts.get("view", 0))
    n_cart = int(counts.get("addtocart", 0))
    n_tx = int(counts.get("transaction", 0))
    return {
        "view": n_view,
        "addtocart": n_cart,
        "transaction": n_tx,
        "cart_per_1000_views": round(n_cart / n_view * 1000, 2) if n_view else float("nan"),
        "tx_per_1000_views": round(n_tx / n_view * 1000, 2) if n_view else float("nan"),
        "tx_per_100_carts": round(n_tx / n_cart * 100, 2) if n_cart else float("nan"),
    }


def session_sensitivity(events: pd.DataFrame, gaps: tuple[int, ...]) -> pd.DataFrame:
    """Session counts and single-event share across gap candidates (Q2).

    Requires ``visitorid`` + ``timestamp`` (epoch ms); delegates ids to
    ``sessionize_events`` so the rule lives in one place.
    """
    rows = []
    for gap in gaps:
        sessioned = sessionize_events(events[["visitorid", "timestamp"]], gap_minutes=gap)
        per_sess = sessioned["session_id"].value_counts()
        rows.append(
            {
                "gap_minutes": gap,
                "sessions": int(sessioned["session_id"].nunique()),
                "single_event_session_pct": round(float((per_sess == 1).mean()) * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("gap_minutes")


def as_of_item_attributes(
    events: pd.DataFrame,
    item_properties: pd.DataFrame,
    properties: list[str] | None = None,
) -> pd.DataFrame:
    """Point-in-time join of property rows onto events (Q3, leakage-safe).

    For each event, attaches the latest property row with
    ``property_time <= event_time`` for the same ``itemid`` (backward
    ``merge_asof``). Rows with no prior property state get NaN plus
    ``asof_covered == False`` — never a forward-filled future value.
    Optionally restricts to ``properties`` (e.g. ``["available"]``).
    A naive latest-value join is deliberately not offered; measure its
    damage with :func:`as_of_vs_naive_disagreement` instead.
    """
    props = item_properties
    if properties is not None:
        props = props[props["property"].isin(properties)]
    props = props.sort_values("timestamp")
    ordered = events.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        ordered,
        props.rename(columns={"timestamp": "prop_time"}),
        left_on="timestamp",
        right_on="prop_time",
        by="itemid",
        direction="backward",
    )
    merged["asof_covered"] = merged["prop_time"].notna()
    return merged


def as_of_vs_naive_disagreement(events: pd.DataFrame, item_properties: pd.DataFrame) -> dict:
    """Quantify naive-vs-as-of join disagreement on one property (Q3).

    Compares the point-in-time state against the latest-known-anytime state
    per item. Returns coverage of each method and their disagreement rate on
    jointly-covered rows. ``item_properties`` must hold a single property
    (filter first with :func:`as_of_item_attributes`' ``properties`` arg).
    """
    asof = as_of_item_attributes(events, item_properties)
    latest = (
        item_properties.sort_values("timestamp").groupby("itemid")["value"].last()
    )
    naive_state = events["itemid"].map(latest)
    both_known = asof["value"].notna() & naive_state.notna()
    return {
        "rows": int(len(events)),
        "asof_coverage_pct": round(float(asof["value"].notna().mean()) * 100, 2),
        "naive_coverage_pct": round(float(naive_state.notna().mean()) * 100, 2),
        "disagreement_pct": round(
            float((asof.loc[both_known, "value"] != naive_state[both_known]).mean()) * 100, 2
        ),
    }


def availability_at_event_time(events: pd.DataFrame, available_props: pd.DataFrame) -> pd.DataFrame:
    """Event mix (percent) by as-of availability state (Q4, associational only).

    Returns a ``state x event`` percentage table over covered rows. Popularity
    confounding is unadjusted here — pair with a within-popularity split
    (notebook Sec 8.1) before treating the gap as a feature signal.
    """
    asof = as_of_item_attributes(events, available_props, properties=["available"])
    covered = asof[asof["value"].notna()]
    return (pd.crosstab(covered["value"], covered["event"], normalize="index") * 100).round(3)


def category_conversion(
    events: pd.DataFrame, item_roots: pd.Series, smooth_m: int = 500
) -> pd.DataFrame:
    """Transaction rate by root category with Empirical-Bayes smoothing (Q5).

    ``item_roots`` maps ``itemid`` to root id (latest-known mapping is an
    association-only caveat — see notebook Sec 7.5). Rates shrink toward the
    global mean with prior weight ``smooth_m`` events so tiny branches cannot
    whiplash. Returns counts plus raw and smoothed rates, sorted descending.
    """
    if smooth_m <= 0:
        raise ValueError("smooth_m must be positive")
    roots = events["itemid"].map(item_roots)
    table = pd.crosstab(roots, events["event"])
    for outcome in ("view", "addtocart", "transaction"):
        if outcome not in table.columns:
            table[outcome] = 0
    table["n"] = table[["view", "addtocart", "transaction"]].sum(axis=1)
    global_rate = (events["event"] == "transaction").mean()
    table["tx_rate_raw_pct"] = table["transaction"] / table["n"] * 100
    table["tx_rate_smooth_pct"] = (
        (table["transaction"] + smooth_m * global_rate) / (table["n"] + smooth_m) * 100
    )
    return table.sort_values("tx_rate_smooth_pct", ascending=False).round(3)


def concentration_curve(values: pd.Series) -> pd.DataFrame:
    """Decile-held share table for a ranked count vector (Q6).

    ``values`` is e.g. events-per-visitor; returns one row per decile (10 =
    top) with population and event shares. Used for head-vs-tail (§6, §8).
    """
    ranked = values.sort_values(ascending=False).to_numpy(dtype=float)
    n = len(ranked)
    edges = np.linspace(0, n, 11).astype(int)
    rows = []
    for d in range(10):
        seg = ranked[edges[d] : edges[d + 1]]
        rows.append(
            {
                "decile": d + 1,
                "population_share_pct": round(len(seg) / n * 100, 2),
                "event_share_pct": round(seg.sum() / ranked.sum() * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("decile")


def property_coverage(item_properties: pd.DataFrame) -> pd.DataFrame:
    """Frequency rank-coverage per property code (Q8).

    One row per ``property`` with row counts, share, and cumulative share —
    the allow-list input for encoding triage (head vs sparse tail).
    """
    counts = item_properties["property"].value_counts()
    total = len(item_properties)
    table = pd.DataFrame(
        {
            "rows": counts,
            "share_pct": (counts / total * 100).round(3),
            "cumulative_share_pct": (counts.cumsum() / total * 100).round(2),
        }
    )
    return table


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion, returned in percent (Stage 6).

    Preferred over the Wald interval at extreme rates (our ~0.8% target).
    Pure stdlib so it runs anywhere.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = p + z**2 / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return (round((center - margin) / denominator * 100, 4),
            round((center + margin) / denominator * 100, 4))


def cramers_v_from_chi2(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """Cramér's V effect size from a chi-square statistic (Stage 6, §13)."""
    if n <= 0 or min(n_rows, n_cols) < 2:
        raise ValueError("need a valid contingency table")
    return round(float(math.sqrt(chi2 / (n * (min(n_rows, n_cols) - 1)))), 3)
