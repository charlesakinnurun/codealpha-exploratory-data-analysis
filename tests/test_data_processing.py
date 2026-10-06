"""Tests for pure helpers + Stage 5 analytical functions (synthetic data only)."""

from pathlib import Path

import pandas as pd

from src.analysis import (
    as_of_item_attributes,
    as_of_vs_naive_disagreement,
    availability_at_event_time,
    category_conversion,
    concentration_curve,
    cramers_v_from_chi2,
    funnel_rates,
    property_coverage,
    session_sensitivity,
    wilson_interval,
)
from src.data_processing import (
    SESSION_GAP_MINUTES_CANDIDATES,
    SESSION_GAP_MINUTES_DEFAULT,
    sessionize_events,
    summarize_duplicates,
    to_utc_datetime,
)


def test_session_gap_default_is_a_candidate():
    assert SESSION_GAP_MINUTES_DEFAULT in SESSION_GAP_MINUTES_CANDIDATES


def test_to_utc_datetime_converts_epoch_millis():
    # Stage 1 anchor: 1430622004384 ms == 2015-05-03 UTC.
    result = to_utc_datetime(pd.Series([1430622004384]))
    assert str(result.iloc[0].date()) == "2015-05-03"
    assert str(result.dt.tz) == "UTC"


def test_raw_inventory_lists_expected_files():
    repo_root = Path(__file__).resolve().parent.parent
    from src.data_processing import list_raw_files

    found_names = {path.name for path in list_raw_files(repo_root)}
    assert {"events.csv", "category_tree.csv"} <= found_names


def _toy_events() -> pd.DataFrame:
    # Two visitors; visitor 1 has a 60-min gap (session split at 30), visitor 2 is continuous.
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2],
            "timestamp": [0, 10 * 60000, 70 * 60000, 0, 5 * 60000],
            "event": ["view", "addtocart", "view", "view", "transaction"],
            "itemid": [7, 7, 8, 9, 9],
        }
    )


def test_sessionize_events_splits_on_gap():
    out = sessionize_events(_toy_events(), gap_minutes=30)
    visitor_1_sessions = out.loc[out["visitorid"] == 1, "session_id"].tolist()
    assert visitor_1_sessions[0] == visitor_1_sessions[1]  # 10-min gap stays
    assert visitor_1_sessions[2] != visitor_1_sessions[1]  # 60-min gap splits
    assert out["session_id"].is_unique is False  # ids shared within session


def test_summarize_duplicates_counts_full_rows_only():
    duped = pd.concat([_toy_events(), _toy_events().head(2)], ignore_index=True)
    summary = summarize_duplicates(duped)
    assert summary["full_row_duplicates"] == 2
    assert summary["rows"] == 7


def test_funnel_rates_conditional_math():
    rates = funnel_rates(_toy_events())
    assert rates["view"] == 3
    assert rates["cart_per_1000_views"] == round(1 / 3 * 1000, 2)
    assert rates["tx_per_100_carts"] == 100.0


def test_session_sensitivity_monotone_in_gap():
    table = session_sensitivity(_toy_events(), gaps=(15, 30, 60))
    assert table.loc[15, "sessions"] >= table.loc[60, "sessions"]


def test_concentration_curve_shares_sum_to_100():
    curve = concentration_curve(pd.Series([10, 5, 3, 1, 1]))
    assert round(curve["event_share_pct"].sum(), 1) == 100.0
    assert curve.loc[10, "event_share_pct"] >= curve.loc[1, "event_share_pct"]


def test_wilson_interval_covers_sample_rate():
    low, high = wilson_interval(22457, 2756101)
    assert low < 22457 / 2756101 * 100 < high
    assert high - low < 0.05  # tight at this n


def test_cramers_v_bounds():
    assert cramers_v_from_chi2(2256.0, 2756101, 11, 2) == round((2256.0 / 2756101) ** 0.5, 3)
    assert 0.0 <= cramers_v_from_chi2(0.0, 100, 3, 2) <= 1.0


def _toy_props() -> pd.DataFrame:
    # item 7: out of stock at t=5, restocked at t=50; item 8: always out.
    return pd.DataFrame(
        {
            "timestamp": [5, 50, 5],
            "itemid": [7, 7, 8],
            "property": ["available", "available", "available"],
            "value": ["0", "1", "0"],
        }
    )


def test_as_of_join_never_uses_future_state():
    events = pd.DataFrame(
        {"timestamp": [10, 60], "itemid": [7, 7], "event": ["view", "view"]}
    )
    out = as_of_item_attributes(events, _toy_props(), properties=["available"])
    assert out["value"].tolist() == ["0", "1"]  # t=10 sees 0, never the t=50 restock
    assert out["asof_covered"].tolist() == [True, True]


def test_as_of_join_marks_uncovered_rows():
    events = pd.DataFrame({"timestamp": [1], "itemid": [7], "event": ["view"]})
    out = as_of_item_attributes(events, _toy_props())
    assert out["asof_covered"].tolist() == [False]
    assert out["value"].isna().all()


def test_naive_vs_asof_disagreement_detects_leak():
    events = pd.DataFrame({"timestamp": [10], "itemid": [7], "event": ["view"]})
    report = as_of_vs_naive_disagreement(events, _toy_props())
    assert report["asof_coverage_pct"] == 100.0
    assert report["disagreement_pct"] == 100.0  # naive says 1, truth-at-time is 0


def test_availability_mix_is_row_normalized():
    events = pd.DataFrame(
        {
            "timestamp": [10, 60, 10],
            "itemid": [7, 7, 8],
            "event": ["view", "transaction", "view"],
        }
    )
    mix = availability_at_event_time(events, _toy_props())
    assert set(mix.index) == {"0", "1"}
    assert ((mix.sum(axis=1) - 100).abs() < 0.01).all()


def test_category_conversion_shrinks_tiny_groups():
    events = pd.DataFrame(
        {
            "itemid": [1, 1, 2, 3],
            "event": ["view", "transaction", "view", "view"],
        }
    )
    table = category_conversion(events, pd.Series({1: "A", 2: "B", 3: "B"}), smooth_m=100)
    assert table.loc["A", "tx_rate_smooth_pct"] > table.loc["B", "tx_rate_smooth_pct"]
    assert table.loc["B", "tx_rate_smooth_pct"] > table.loc["B", "tx_rate_raw_pct"]


def test_property_coverage_cumsum_ends_at_100():
    table = property_coverage(
        pd.DataFrame({"property": ["a", "a", "a", "b", "c"]})
    )
    assert table.loc["a", "share_pct"] == 60.0
    assert table["cumulative_share_pct"].iloc[-1] == 100.0
