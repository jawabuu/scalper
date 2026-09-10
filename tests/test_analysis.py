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


# ── Internal consistency ─────────────────────────────────────────────────────

def test_reconciliation_closes_when_the_numbers_agree():
    """realised - fees must equal the wallet change."""
    from bot.analysis import reconcile
    trades = [{"realised_pnl_usdt": 0.0, "fees_usdt": 2.07, "pnl_source": "ledger"},
              {"realised_pnl_usdt": 12.5, "fees_usdt": 2.10, "pnl_source": "ledger"},
              {"realised_pnl_usdt": -24.3, "fees_usdt": 1.95, "pnl_source": "ledger"}]
    r = reconcile(trades, wallet_now=5000.0 - 17.92, wallet_start=5000.0)
    assert r["net_pnl"] == pytest.approx(-17.92, abs=0.01)
    assert r["reconciles"] is True
    assert r["discrepancy"] == pytest.approx(0.0, abs=0.01)


def test_reconciliation_flags_a_gap():
    from bot.analysis import reconcile
    trades = [{"realised_pnl_usdt": 60.0, "fees_usdt": 0.0, "pnl_source": "computed"}]
    r = reconcile(trades, wallet_now=4980.0, wallet_start=5000.0)
    assert r["reconciles"] is False
    assert r["discrepancy"] == pytest.approx(-80.0)


def test_reconciliation_counts_estimated_exits():
    from bot.analysis import reconcile
    trades = [{"realised_pnl_usdt": 1.0, "pnl_source": "computed"},
              {"realised_pnl_usdt": 1.0, "pnl_source": "ledger"}]
    assert reconcile(trades, None, None)["estimated_exits"] == 1


def test_roi_mismatch_is_reported():
    """
    A price-derived ROI that disagrees with the money received means the
    trade's own numbers are inconsistent — SOLV said +4.18% against a true 0.
    """
    trades = [{"side": "short", "final_roi": 4.18, "roi_from_realised": 0.0,
               "peak_roi": 4.18, "realised_pnl_usdt": 0.0, "entry_context": {}}]
    r = analyse(trades)
    assert r["roi_mismatches"] == 1
    assert any("disagrees with the ROI implied" in n for n in r["notes"])


def test_no_mismatch_when_consistent():
    trades = [{"side": "short", "final_roi": 4.18, "roi_from_realised": 4.18,
               "peak_roi": 4.18, "realised_pnl_usdt": 4.32, "entry_context": {}}]
    assert analyse(trades)["roi_mismatches"] == 0


def test_wallet_reconciles_with_full_round_trip_fees():
    """
    With only the exit side captured the wallet was short by exactly the
    missing entry commission.
    """
    from bot.analysis import reconcile
    trades = [{"realised_pnl_usdt": 15.3342, "fees_usdt": 2.60, "pnl_source": "ledger"},
              {"realised_pnl_usdt": -24.5864, "fees_usdt": 2.52, "pnl_source": "ledger"},
              {"realised_pnl_usdt": -28.6656, "fees_usdt": 2.66, "pnl_source": "ledger"},
              {"realised_pnl_usdt": -25.2470, "fees_usdt": 2.62, "pnl_source": "ledger"}]
    r = reconcile(trades, wallet_now=4926.43, wallet_start=5000.0)
    assert r["reconciles"] is True
    assert abs(r["discrepancy"]) < 0.05


# ── Return on capital, not a mean of percentages ─────────────────────────────

def _sized(roi, pnl, margin):
    return {"side": "short", "final_roi": roi, "peak_roi": max(roi, 0) + 3,
            "realised_pnl_usdt": pnl, "margin": margin, "entry_context": {}}


def test_return_on_capital_weights_by_size():
    """
    Averaging ROI percentages treats a large trade the same as a small one. One
    session showed +8.32% as a simple mean while the capital actually returned
    +2.80% — the single loss carried more than double the margin of either win.
    """
    trades = [_sized(17.47, 14.4273, 82.58),
              _sized(-13.70, -24.9701, 182.26),
              _sized(21.20, 20.6907, 97.60)]
    o = analyse(trades)["overall"]
    assert o["avg_roi"] == pytest.approx(8.32, abs=0.01)
    assert o["capital_deployed"] == pytest.approx(362.44, abs=0.01)
    assert o["return_on_capital"] == pytest.approx(2.80, abs=0.05)


def test_return_on_capital_is_net_when_fees_are_known():
    trades = [dict(_sized(17.47, 14.4273, 82.58), fees_usdt=2.05,
                   net_pnl_usdt=12.3773),
              dict(_sized(-13.70, -24.9701, 182.26), fees_usdt=2.05,
                   net_pnl_usdt=-27.0201),
              dict(_sized(21.20, 20.6907, 97.60), fees_usdt=2.04,
                   net_pnl_usdt=18.6507)]
    assert analyse(trades)["overall"]["return_on_capital"] == pytest.approx(1.11, abs=0.05)


def test_margin_absent_and_underivable_stays_unreported():
    """
    With no margin AND no ROI there is nothing to derive from, so the figure
    is omitted rather than guessed. (Where ROI exists it IS derived — see
    test_margin_is_derived_when_not_recorded.)
    """
    trades = [{"side": "short", "final_roi": 0.0, "peak_roi": 0.0,
               "realised_pnl_usdt": 0.0, "entry_context": {}}]
    o = analyse(trades)["overall"]
    assert o["return_on_capital"] is None
    assert o["capital_deployed"] is None


def test_equal_sizes_make_the_two_measures_agree():
    trades = [_sized(10.0, 10.0, 100.0), _sized(-6.0, -6.0, 100.0)]
    o = analyse(trades)["overall"]
    assert o["avg_roi"] == pytest.approx(2.0)
    assert o["return_on_capital"] == pytest.approx(2.0)


# ── Time-of-day splits ───────────────────────────────────────────────────────

def _at_hour(side, roi, pnl, margin, hour):
    from datetime import datetime, timezone
    ts = datetime(2026, 9, 7, hour, 30, tzinfo=timezone.utc).timestamp()
    return {"side": side, "final_roi": roi, "peak_roi": max(roi, 0) + 3,
            "realised_pnl_usdt": pnl, "margin": margin, "opened_at": ts,
            "entry_context": {}}


def _regime_set():
    """Shorts win overnight, longs win in the US session."""
    t = []
    for h in (2, 4, 6):
        t += [_at_hour("short", 18, 20, 100, h), _at_hour("long", -12, -24, 120, h)]
    for h in (18, 20, 22):
        t += [_at_hour("short", -11, -22, 110, h), _at_hour("long", 16, 18, 95, h)]
    return t


def test_sessions_separate_direction_performance():
    """
    Comparing directions WITHIN the same hours controls for the regime —
    unlike long-vs-short across a whole run, where one side may simply have
    matched the trend.
    """
    r = analyse(_regime_set())
    by = {b["label"]: b for b in r["by_session"] if b["n"]}
    assert by["Asia"]["short"]["win_rate"] == 100.0
    assert by["Asia"]["long"]["win_rate"] == 0.0
    assert by["US"]["long"]["win_rate"] == 100.0
    assert by["US"]["short"]["win_rate"] == 0.0


def test_hour_buckets_only_include_hours_with_trades():
    r = analyse(_regime_set())
    assert len(r["by_hour"]) == 6
    assert all(b["n"] for b in r["by_hour"])


def test_entry_time_is_used_not_exit_time():
    """A trade opened at 02:00 and closed at 09:00 belongs to the Asia session."""
    from datetime import datetime, timezone
    t = [{"side": "short", "final_roi": 5.0, "peak_roi": 8.0,
          "realised_pnl_usdt": 5.0, "margin": 100.0,
          "opened_at": datetime(2026, 9, 7, 2, 0, tzinfo=timezone.utc).timestamp(),
          "closed_at": datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc).timestamp(),
          "entry_context": {}}]
    by = {b["label"]: b for b in analyse(t)["by_session"] if b["n"]}
    assert "Asia" in by and "Europe" not in by


def test_trades_without_timestamps_are_skipped():
    t = [{"side": "short", "final_roi": 5.0, "peak_roi": 8.0,
          "realised_pnl_usdt": 5.0, "entry_context": {}}]
    assert all(not b["n"] for b in analyse(t)["by_session"])


def test_every_session_carries_a_confidence_label():
    r = analyse(_regime_set())
    for b in r["by_session"]:
        if b["n"]:
            assert b["confidence"] in ("insufficient", "thin", "usable")


# ── Margin must be usable on older records too ───────────────────────────────

def test_margin_is_derived_when_not_recorded():
    """
    Trades closed before margin was stored showed a dash for return on
    capital. It is recoverable from the two fields that ARE present:
    margin = realised / (ROI / 100).
    """
    from bot.analysis import _margin
    t = {"realised_pnl_usdt": 14.4273, "final_roi": 17.47}
    assert _margin(t) == pytest.approx(82.58, abs=0.05)


def test_recorded_margin_wins_over_the_derivation():
    from bot.analysis import _margin
    t = {"margin": 100.0, "realised_pnl_usdt": 14.4273, "final_roi": 17.47}
    assert _margin(t) == pytest.approx(100.0)


def test_derivation_prefers_roi_from_realised():
    """That ROI comes from the same figure as the money, so it cannot disagree."""
    from bot.analysis import _margin
    t = {"realised_pnl_usdt": 10.0, "final_roi": 50.0, "roi_from_realised": 10.0}
    assert _margin(t) == pytest.approx(100.0)


def test_zero_roi_cannot_be_derived():
    from bot.analysis import _margin
    assert _margin({"realised_pnl_usdt": 0.0, "final_roi": 0.0}) is None


def test_return_on_capital_works_without_recorded_margin():
    trades = [{"side": "short", "final_roi": 17.47, "peak_roi": 20.29,
               "realised_pnl_usdt": 14.4273, "entry_context": {}},
              {"side": "short", "final_roi": -13.7, "peak_roi": 1.28,
               "realised_pnl_usdt": -1.2573, "entry_context": {}}]
    o = analyse(trades)["overall"]
    assert o["return_on_capital"] is not None
    assert o["capital_deployed"] is not None


# ── Invented P&L must not pollute the totals ─────────────────────────────────

def _verified_trade(sym, roi, pnl, margin):
    return {"symbol": sym, "side": "long", "final_roi": roi, "peak_roi": roi + 2,
            "realised_pnl_usdt": pnl, "margin": margin, "pnl_source": "ledger",
            "pnl_verified": True, "entry_context": {}}


def _invented_trade(sym, roi, pnl):
    return {"symbol": sym, "side": "short", "final_roi": roi, "peak_roi": 0.0,
            "realised_pnl_usdt": pnl, "pnl_source": "computed",
            "pnl_verified": False, "exit_is_estimate": True,
            "entry_context": {}}


def test_unverified_trades_are_excluded_and_counted():
    """
    With neither the ledger nor the fills readable, the exit was reconstructed
    from a guess: one trade reported -75% ROI on a position whose stop was
    capped at -30%, another a flat zero. Averaging those corrupts every total.
    """
    trades = [_verified_trade("COLLECT", 34.27, 29.3911, 85.8),
              _invented_trade("BTR", -75.12, -64.6411),
              _invented_trade("ZORA", 0.0, 0.0)]
    r = analyse(trades)
    assert r["overall"]["n"] == 1
    assert r["unverified_trades"] == 2
    assert set(r["unverified_symbols"]) == {"BTR", "ZORA"}
    assert any("EXCLUDED" in n for n in r["notes"])


def test_excluding_them_lets_the_totals_mean_something():
    trades = [_verified_trade("A", 10.0, 10.0, 100.0),
              _invented_trade("B", -75.0, -64.0)]
    o = analyse(trades)["overall"]
    assert o["total_realised"] == pytest.approx(10.0)
    assert o["win_rate"] == pytest.approx(100.0)


def test_older_records_without_a_flag_use_the_estimate_marker():
    from bot.analysis import _verified
    assert _verified({"exit_is_estimate": True}) is False
    assert _verified({"exit_is_estimate": False}) is True
    assert _verified({"pnl_source": "ledger"}) is True
    assert _verified({"pnl_source": "computed"}) is False


def test_no_verified_trades_reports_zero_not_a_crash():
    r = analyse([_invented_trade("B", -75.0, -64.0)])
    assert r["overall"]["n"] == 0
    assert r["unverified_trades"] == 1


# ── Account return ───────────────────────────────────────────────────────────

def _ar_trade(pnl, net, when, verified=True):
    return {"symbol": "X", "realised_pnl_usdt": pnl, "net_pnl_usdt": net,
            "fees_usdt": round(pnl - net, 4), "pnl_verified": verified,
            "closed_at": when}


def test_account_return_divides_by_the_starting_balance():
    """
    Distinct from return on capital, which divides by the SUM of margins across
    sequential trades — the same money recycled, so that is return per unit of
    turnover rather than growth of the account.
    """
    from bot.analysis import account_return
    import time
    now = time.time()
    r = account_return([_ar_trade(29.39, 26.80, now - 7200),
                        _ar_trade(-25.26, -27.85, now - 600)],
                       baseline=5000.0)
    assert r["net_pnl"] == pytest.approx(-1.05, abs=0.01)
    assert r["pct"] == pytest.approx(-0.021, abs=0.001)


def test_today_is_reported_separately():
    from bot.analysis import account_return
    import time
    now = time.time()
    r = account_return([_ar_trade(29.39, 26.80, now - 7200),
                        _ar_trade(-25.26, -27.85, now - 600)],
                       baseline=5000.0, day_baseline=5000.0,
                       day_start_ts=now - 3600)
    assert r["day_trades"] == 1
    assert r["day_net_pnl"] == pytest.approx(-27.85, abs=0.01)


def test_unverified_trades_are_excluded_from_account_return():
    from bot.analysis import account_return
    import time
    now = time.time()
    r = account_return([_ar_trade(10.0, 8.0, now),
                        _ar_trade(-64.64, -64.64, now, verified=False)],
                       baseline=5000.0)
    assert r["trades"] == 1
    assert r["net_pnl"] == pytest.approx(8.0)


def test_disagreement_with_the_wallet_is_flagged():
    """
    Excluded trades mean the record can disagree with the balance. Saying so
    beats showing a figure that quietly contradicts the wallet above it.
    """
    from bot.analysis import account_return
    import time
    now = time.time()
    r = account_return([_ar_trade(10.0, 8.0, now),
                        _ar_trade(-64.64, -64.64, now, verified=False)],
                       baseline=5000.0, wallet_now=5029.66)
    assert r["disagrees_with_wallet"] is True
    assert r["wallet_gap"] == pytest.approx(21.66, abs=0.01)


def test_agreement_is_not_flagged():
    from bot.analysis import account_return
    import time
    now = time.time()
    r = account_return([_ar_trade(10.0, 8.0, now)],
                       baseline=5000.0, wallet_now=5008.0)
    assert r["disagrees_with_wallet"] is False


def test_no_baseline_yields_no_percentage():
    from bot.analysis import account_return
    r = account_return([], baseline=None)
    assert r["pct"] is None


# ── Regime splits (measurement only) ─────────────────────────────────────────

def _regime_trade(side, roi, pnl, breadth, htf):
    import time
    now = time.time()
    return {"symbol": "X", "side": side, "final_roi": roi,
            "peak_roi": max(roi, 0) + 3, "realised_pnl_usdt": pnl,
            "margin": 100.0, "pnl_verified": True,
            "opened_at": now, "closed_at": now,
            "entry_context": {"breadth_pct": breadth, "htf_trend_pct": htf}}


def _regime_trades():
    """Shorts win while the market falls; longs win while it rises."""
    return [_regime_trade("short", 18, 18, 20, -1.2),
            _regime_trade("short", 16, 16, 25, -0.9),
            _regime_trade("long", -12, -12, 22, -1.1),
            _regime_trade("long", 15, 15, 80, 1.4),
            _regime_trade("long", 13, 13, 75, 1.1),
            _regime_trade("short", -11, -11, 78, 1.0)]


def test_breadth_split_separates_directions():
    """
    Breadth is a DIRECT measure of regime — the share of the scanned universe
    up on 24h — where session is only a proxy for it.
    """
    r = analyse(_regime_trades())
    by = {b["label"]: b for b in r["by_breadth"] if b["n"]}
    assert by["<30% up"]["short"]["win_rate"] == 100.0
    assert by["<30% up"]["long"]["win_rate"] == 0.0
    assert by[">70% up"]["long"]["win_rate"] == 100.0
    assert by[">70% up"]["short"]["win_rate"] == 0.0


def test_htf_alignment_split():
    r = analyse(_regime_trades())
    by = {b["label"]: b for b in r["by_htf_alignment"] if b["n"]}
    assert by["with the hourly trend"]["n"] == 4
    assert by["with the hourly trend"]["win_rate"] == 100.0
    assert by["against the hourly trend"]["n"] == 2
    assert by["against the hourly trend"]["win_rate"] == 0.0


def test_alignment_is_direction_aware():
    """A negative hourly trend is WITH a short and AGAINST a long."""
    r = analyse([_regime_trade("short", 5, 5, 40, -1.0),
                 _regime_trade("long", 5, 5, 40, -1.0)])
    by = {b["label"]: b for b in r["by_htf_alignment"] if b["n"]}
    assert by["with the hourly trend"]["short"]["n"] == 1
    assert by["against the hourly trend"]["long"]["n"] == 1


def test_trades_without_regime_context_are_skipped():
    t = [{"symbol": "X", "side": "short", "final_roi": 5.0, "peak_roi": 8.0,
          "realised_pnl_usdt": 5.0, "pnl_verified": True, "entry_context": {}}]
    r = analyse(t)
    assert all(not b["n"] for b in r["by_breadth"])
    assert all(not b["n"] for b in r["by_htf_alignment"])


def test_regime_splits_carry_confidence_labels():
    r = analyse(_regime_trades())
    for b in r["by_breadth"] + r["by_htf_alignment"]:
        if b["n"]:
            assert b["confidence"] in ("insufficient", "thin", "usable")


# ── Fail-fast impact (measurement only) ──────────────────────────────────────

def _ff_winner(roi, pnl, secs):
    return {"side": "short", "final_roi": roi, "peak_roi": roi + 3,
            "realised_pnl_usdt": pnl, "margin": 100.0, "pnl_verified": True,
            "secs_to_first_positive": secs, "entry_context": {}}


def _ff_loser(final, at60, at180, at300):
    return {"side": "short", "final_roi": final, "peak_roi": 0.0,
            "realised_pnl_usdt": final, "margin": 100.0, "pnl_verified": True,
            "secs_to_first_positive": None, "roi_at_60s": at60,
            "roi_at_180s": at180, "roi_at_300s": at300, "entry_context": {}}


def _ff_set():
    return [_ff_winner(20, 20, 15), _ff_winner(18, 18, 45),
            _ff_winner(25, 25, 240), _ff_winner(30, 30, 400),
            _ff_loser(-25, -4, -11, -19), _ff_loser(-24, -6, -14, -21),
            _ff_loser(-25, -3, -9, -17)]


def test_cutoff_counts_the_winners_it_would_have_killed():
    """
    A winner that dipped for four minutes before running is indistinguishable
    from one that went green immediately unless the timing is recorded — so
    the cost of any cutoff would otherwise be unmeasurable.
    """
    r = {f["cutoff_s"]: f for f in analyse(_ff_set())["fail_fast_impact"]}
    assert r[60]["winners_cut"] == 2      # the 240s and 400s winners
    assert r[300]["winners_cut"] == 1     # only the 400s one
    assert r[60]["of_winners"] == 4


def test_cutoff_values_the_cost():
    r = {f["cutoff_s"]: f for f in analyse(_ff_set())["fail_fast_impact"]}
    assert r[60]["winner_value_lost"] == pytest.approx(55.0)
    assert r[300]["winner_value_lost"] == pytest.approx(30.0)


def test_cutoff_shows_what_it_would_have_saved():
    """Where a never-green loser stood at the cutoff versus where it ended."""
    r = {f["cutoff_s"]: f for f in analyse(_ff_set())["fail_fast_impact"]}
    assert r[60]["losers_never_green"] == 3
    assert r[60]["avg_roi_at_cutoff"] == pytest.approx(-4.33, abs=0.05)
    assert r[60]["avg_roi_saved"] > r[300]["avg_roi_saved"]


def test_later_cutoffs_save_less():
    """The longer you wait, the more of the loss has already happened."""
    r = {f["cutoff_s"]: f for f in analyse(_ff_set())["fail_fast_impact"]}
    assert r[60]["avg_roi_saved"] > r[180]["avg_roi_saved"] > r[300]["avg_roi_saved"]


def test_no_timing_data_yields_empty_counts():
    t = [{"side": "short", "final_roi": 5.0, "peak_roi": 8.0,
          "realised_pnl_usdt": 5.0, "pnl_verified": True, "entry_context": {}}]
    for f in analyse(t)["fail_fast_impact"]:
        assert f["of_winners"] == 0
