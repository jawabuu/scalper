"""
tools/compare_callback_mult.py — the before/after split for a multiplier change.

Exists because every earlier comparison was CROSS-CONTAINER, and demo is live
data with a ~0.751 factor on ATR plus a different leverage and symbol set.
Those confounds are larger than the effect being measured.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_callback_mult as cc          # noqa: E402


def _trade(mult=0.75, atr=1.0, lev=20.0, final_roi=10.0, source="atr_floor",
           peak_roi=5.0, **kw):
    t = {
        "leverage": lev, "final_roi": final_roi, "roi_at_first_sight": -1.0,
        "peak_roi": peak_roi,
        "drift_since_sizing_pct": -0.3, "net_pnl_usdt": 5.0, "fees_usdt": 1.0,
        "exit_reason": "trail", "opened_at": 1_000_000.0, "symbol": "X/USDT:USDT",
        "entry_context": {"atr_pct": atr, "callback_pct": atr * mult,
                          "callback_source": source},
    }
    t.update(kw)
    return t


def test_non_floor_trades_are_KEPT_but_not_given_a_multiplier():
    """
    The first run invented cohorts at 0.70/0.90/1.00 that were nothing but
    ratio-sourced trades whose callback/atr happened to land there — so they
    must not be grouped BY multiplier. But dropping them hid the change's
    second effect (live 09-21: floor-binding fell 97% -> 52%), so they are
    kept as their own labelled cohort instead.
    """
    rows = cc.enrich([_trade(source="atr_floor"), _trade(source="ratio"),
                      _trade(source="")])
    assert len(rows) == 3
    assert [r["mult"] is None for r in rows] == [False, True, True]
    assert {r["src"] for r in rows} == {"atr_floor", "ratio", "unknown"}


def test_everything_is_in_PRICE_percent_not_ROI():
    """
    ROI multiplies by leverage. Comparing ROI across any leverage change
    silently scales the result — the error that produced four wrong
    conclusions in this investigation.
    """
    a = cc.enrich([_trade(lev=10.0, final_roi=10.0)])[0]
    b = cc.enrich([_trade(lev=20.0, final_roi=20.0)])[0]
    assert a["final_px"] == b["final_px"] == 1.0


def test_the_multiplier_is_derived_from_the_trade_not_a_timestamp():
    # No deploy time to remember; a re-run months later still splits right.
    rows = cc.enrich([_trade(mult=0.75, atr=2.0), _trade(mult=1.25, atr=0.5)])
    assert sorted(r["mult"] for r in rows) == [0.75, 1.25]


def test_a_single_multiplier_is_reported_as_a_BASELINE_not_a_comparison(capsys, tmp_path):
    j = tmp_path / "trades.jsonl"
    j.write_text("\n".join(json.dumps(_trade()) for _ in range(40)))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "only ONE multiplier" in out and "BEFORE baseline" in out


def test_a_small_group_is_called_out_loudly(capsys, tmp_path):
    j = tmp_path / "trades.jsonl"
    rows = [_trade(mult=0.75) for _ in range(40)] + [_trade(mult=1.25) for _ in range(4)]
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "GROUPS UNDER" in out and "Do not act on this output" in out


def test_a_leverage_difference_between_groups_is_called_out(capsys, tmp_path):
    # Then it is not a within-instance comparison any more.
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.75, lev=20.0) for _ in range(35)]
            + [_trade(mult=1.25, lev=10.0) for _ in range(35)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    assert "LEVERAGE DIFFERS" in capsys.readouterr().out


def test_a_market_regime_difference_is_called_out(capsys, tmp_path):
    # A difference in outcome may be the market, not the multiplier.
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.75, atr=0.5) for _ in range(35)]
            + [_trade(mult=1.25, atr=2.0) for _ in range(35)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    assert "MARKET REGIME DIFFERS" in capsys.readouterr().out


def test_never_green_is_reported_and_split_by_outcome(capsys, tmp_path):
    """
    On the 2026-09-21 exports never_green split winners from losers more
    cleanly than anything else: 30% of live trades, median -0.570% of price,
    against +0.237% for those that went green. A wider callback fills deeper
    into the bounce, so lowering the multiplier should REDUCE this rate —
    the sharpest prediction of the change.
    """
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.75, peak_roi=5.0, final_roi=8.0) for _ in range(30)]
            + [_trade(mult=0.75, peak_roi=-3.0, final_roi=-12.0) for _ in range(10)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "NEVER_GREEN" in out
    assert "25%" in out                      # 10 of 40


def test_never_green_is_None_when_peak_is_missing():
    # Absent must not be counted as "never green".
    rows = cc.enrich([_trade(peak_roi=None)])
    assert rows[0]["never_green"] is None


def test_ratio_sourced_trades_are_their_OWN_cohort_not_dropped(capsys, tmp_path):
    """
    The callback is max(ratio-derived, ATR floor). Lowering the floor from
    1.25 to 0.563 on live took floor-binding from 97% to 52% — eleven trades
    vanished from the comparison, and a result read as "0.563 vs 1.25" would
    partly have been "ratio-source vs floor-source".
    """
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.56) for _ in range(12)]
            + [_trade(mult=1.25) for _ in range(35)]
            + [_trade(source="ratio") for _ in range(11)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "source 'ratio': 11 trades" in out
    assert "ratio" in out.split("FINAL OUTCOME")[1]


def test_a_non_floor_cohort_never_gets_a_numeric_multiplier():
    # A number there would be read as a setting when it is an accident.
    rows = cc.enrich([_trade(source="ratio", mult=0.9)])
    assert rows[0]["mult"] is None and rows[0]["src"] == "ratio"


def test_an_ATR_SELECTED_cohort_is_called_out(capsys, tmp_path):
    """
    The floor/ratio split is endogenous: the floor binds when ATR is high, so
    'ratio' is largely a label for calm coins. Live 2026-09-22 showed ratio at
    median ATR 0.556% vs ~0.95% for the floor groups, and it looked best on
    every outcome.
    """
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.56, atr=1.0) for _ in range(12)]
            + [_trade(mult=1.25, atr=1.0) for _ in range(35)]
            + [_trade(source="ratio", atr=0.5) for _ in range(11)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "ATR-SELECTED" in out
    assert "fair comparison" in out


def test_the_regime_check_STILL_FIRES_when_a_third_cohort_exists(capsys, tmp_path):
    """
    The first version guarded with `if len(keys) == 2`, so the moment a
    'ratio' cohort appeared the regime check silently stopped running — and
    it stopped precisely when the floor groups had drifted to 1.29 vs 1.01
    median ATR. A guard that switches itself off as the data gets more
    complex is worse than no guard.
    """
    j = tmp_path / "trades.jsonl"
    rows = ([_trade(mult=0.60, atr=1.29) for _ in range(34)]
            + [_trade(mult=1.25, atr=1.01) for _ in range(37)]
            + [_trade(source="ratio", atr=0.67) for _ in range(22)])
    j.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--journal", str(j)]
    cc.main()
    out = capsys.readouterr().out
    assert "MARKET REGIME DIFFERS between 0.60 and 1.20" in out or \
           "MARKET REGIME DIFFERS between 1.20 and 0.60" in out


def test_a_missing_journal_is_not_a_crash(capsys, tmp_path):
    sys.argv = ["x", "--journal", str(tmp_path / "nope.jsonl")]
    assert cc.main() == 1
