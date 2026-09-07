"""Tests for trade analysis — especially that it refuses to overstate thin data."""
import pytest

from bot.analysis import (
    analyse, group_stats, confidence_for, bucket_by, Bucket,
    MIN_USABLE, MIN_INDICATIVE,
)


def _t(roi, peak=None, realised=None, side="short", rsi=75.0,
       dist=2.0, atr=0.6, reentry=False, reason="trail"):
    return {
        "side": side, "final_roi": roi,
        "peak_roi": peak if peak is not None else max(roi, 0) + 5,
        "realised_pnl_usdt": realised if realised is not None else roi * 0.1,
        "exit_reason": reason,
        "entry_context": {"rsi": rsi, "dist_to_extreme_pct": dist,
                          "atr_pct": atr, "was_reentry": reentry},
    }


# ── Sample-size honesty ──────────────────────────────────────────────────────

def test_confidence_labels():
    assert confidence_for(3) == "insufficient"
    assert confidence_for(MIN_INDICATIVE) == "thin"
    assert confidence_for(MIN_USABLE) == "usable"


def test_tiny_sample_is_flagged_not_characterised():
    r = analyse([_t(10), _t(12), _t(-4)])
    assert r["overall"]["confidence"] == "insufficient"
    assert any("Nothing here supports a conclusion" in n for n in r["notes"])


def test_moderate_sample_warns_splits_are_thinner():
    r = analyse([_t(5) for _ in range(20)])
    assert any("not enough to act on" in n for n in r["notes"])


def test_empty_history_does_not_crash():
    r = analyse([])
    assert r["overall"]["n"] == 0
    assert r["by_rsi"] and all(b["n"] == 0 for b in r["by_rsi"])


# ── Expectancy is the headline, not win rate ─────────────────────────────────

def test_high_win_rate_with_negative_expectancy_is_called_out():
    """Small frequent wins with rare large losses must not read as success."""
    trades = [_t(2, realised=0.2) for _ in range(8)] + [_t(-30, realised=-3.0) for _ in range(4)]
    r = analyse(trades)
    assert r["overall"]["win_rate"] >= 50
    assert r["overall"]["expectancy_usdt"] < 0
    assert any("exit problem" in n for n in r["notes"])


def test_giveback_is_surfaced_when_large():
    trades = [_t(5, peak=30) for _ in range(12)]
    r = analyse(trades)
    assert r["overall"]["avg_giveback"] == pytest.approx(25.0)
    assert any("give-back" in n for n in r["notes"])


# ── Bucketing ────────────────────────────────────────────────────────────────

def test_rsi_buckets_split_correctly():
    trades = [_t(5, rsi=65), _t(5, rsi=72), _t(5, rsi=80), _t(5, rsi=90)]
    got = {b["label"]: b["n"] for b in analyse(trades)["by_rsi"] if b["n"]}
    assert got == {"60-70": 1, "70-78": 1, "78-85": 1, "85+": 1}


def test_distance_buckets_split_correctly():
    trades = [_t(5, dist=0.5), _t(5, dist=1.5), _t(5, dist=2.5), _t(5, dist=6.0)]
    got = {b["label"]: b["n"] for b in analyse(trades)["by_distance_to_extreme"] if b["n"]}
    assert got == {"<1%": 1, "1-2%": 1, "2-3%": 1, "5%+": 1}


def test_missing_entry_context_is_excluded_not_guessed():
    trades = [_t(5), {"side": "short", "final_roi": 5.0, "peak_roi": 8.0}]
    r = analyse(trades)
    assert r["overall"]["n"] == 2                 # both counted overall
    assert sum(b["n"] for b in r["by_rsi"]) == 1  # only the stamped one bucketed


def test_bucket_boundaries_are_half_open():
    b = Bucket("x", 1.0, 2.0)
    assert b.holds(1.0) and not b.holds(2.0)
    assert not b.holds(None)


# ── Splits that answer the open questions ────────────────────────────────────

def test_reentries_compared_against_fresh_entries():
    trades = ([_t(-5, reentry=True) for _ in range(4)]
              + [_t(8, reentry=False) for _ in range(10)])
    r = analyse(trades)
    assert r["reentries"]["override_reentries"]["n"] == 4
    assert r["reentries"]["fresh_entries"]["n"] == 10
    assert r["reentries"]["override_reentries"]["avg_roi"] < 0


def test_exit_reasons_are_broken_out():
    trades = [_t(9, reason="trail"), _t(-10, reason="stop"), _t(-1, reason="timeout")]
    labels = {b["label"] for b in analyse(trades)["by_exit_reason"]}
    assert labels == {"trail", "stop", "timeout"}


def test_long_and_short_reported_separately():
    r = analyse([_t(5, side="long"), _t(-3, side="short"), _t(7, side="short")])
    assert r["by_side"]["long"]["n"] == 1
    assert r["by_side"]["short"]["n"] == 2


# ── Trough / stop-impact reporting ───────────────────────────────────────────

def _w(roi, peak, trough):
    return {"side": "long", "final_roi": roi, "peak_roi": peak,
            "trough_roi": trough, "realised_pnl_usdt": roi * 0.25,
            "exit_reason": "trail" if roi > 0 else "stop", "entry_context": {}}


def test_stop_impact_counts_winners_that_dipped_past_each_level():
    """
    Peak alone cannot say whether a tighter stop would have cut a winner short.
    The trough answers it: a winner that dipped to -11% would have been stopped
    out by a 5% or 10% stop but survived 15%.
    """
    trades = [_w(29.0, 41.0, -3.2), _w(8.2, 13.3, -11.4), _w(24.2, 30.3, -16.2)]
    si = {r["stop_roi"]: r["winners_cut"] for r in analyse(trades)["stop_impact"]}
    # -3.2% never reaches a -5% stop, so only the -11.4 and -16.2 are cut.
    assert si[5.0] == 2
    assert si[10.0] == 2
    assert si[15.0] == 1
    assert si[20.0] == 0


def test_stop_impact_ignores_losers():
    """A loser was stopped out anyway — only winners answer the question."""
    trades = [_w(20.0, 25.0, -2.0), _w(-13.0, 0.0, -13.0)]
    rows = analyse(trades)["stop_impact"]
    assert all(r["of_winners"] == 1 for r in rows)


def test_stop_impact_carries_a_confidence_label():
    trades = [_w(20.0, 25.0, -2.0)]
    assert all(r["confidence"] == "insufficient"
               for r in analyse(trades)["stop_impact"])


def test_trough_note_explains_an_empty_dataset():
    r = analyse([{"side": "long", "final_roi": 5.0, "peak_roi": 8.0}])
    assert "No trough data yet" in r["trough_note"]


def test_trough_note_reports_the_deepest_dip():
    trades = [_w(29.0, 41.0, -3.2), _w(24.2, 30.3, -16.2)]
    assert "-16.2" in analyse(trades)["trough_note"]


# ── Fees must not be invisible ───────────────────────────────────────────────

def test_expectancy_prefers_the_net_figure():
    """
    Binance's realizedPnl excludes commission, so a gross expectancy reads
    positive while the wallet falls — +60 USDT of P&L against a 20 USDT wallet
    loss, the difference being 80 USDT of fees.
    """
    trades = [{"side": "long", "final_roi": 5.0, "peak_roi": 8.0,
               "realised_pnl_usdt": 6.0, "fees_usdt": 4.0,
               "net_pnl_usdt": 2.0, "entry_context": {}} for _ in range(10)]
    r = analyse(trades)
    assert r["overall"]["expectancy_usdt"] == pytest.approx(2.0)
    assert r["overall"]["total_fees"] == pytest.approx(40.0)


def test_falls_back_to_gross_when_net_is_absent():
    trades = [{"side": "long", "final_roi": 5.0, "peak_roi": 8.0,
               "realised_pnl_usdt": 6.0, "entry_context": {}} for _ in range(10)]
    assert analyse(trades)["overall"]["expectancy_usdt"] == pytest.approx(6.0)


def test_fee_note_explains_the_discrepancy():
    trades = [{"side": "long", "final_roi": 5.0, "peak_roi": 8.0,
               "realised_pnl_usdt": 6.0, "fees_usdt": 4.0,
               "net_pnl_usdt": 2.0, "entry_context": {}} for _ in range(12)]
    notes = " ".join(analyse(trades)["notes"])
    assert "Fees so far" in notes and "gross" in notes


def test_no_fee_note_when_fees_are_unknown():
    trades = [{"side": "long", "final_roi": 5.0, "peak_roi": 8.0,
               "realised_pnl_usdt": 6.0, "entry_context": {}} for _ in range(12)]
    assert not any("Fees so far" in n for n in analyse(trades)["notes"])
