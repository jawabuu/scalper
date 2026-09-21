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
           **kw):
    t = {
        "leverage": lev, "final_roi": final_roi, "roi_at_first_sight": -1.0,
        "drift_since_sizing_pct": -0.3, "net_pnl_usdt": 5.0, "fees_usdt": 1.0,
        "exit_reason": "trail", "opened_at": 1_000_000.0, "symbol": "X/USDT:USDT",
        "entry_context": {"atr_pct": atr, "callback_pct": atr * mult,
                          "callback_source": source},
    }
    t.update(kw)
    return t


def test_only_atr_floor_trades_are_grouped():
    """
    The first run invented cohorts at 0.70, 0.90 and 1.00 that were nothing but
    AUTO_CALLBACK_RATIO-sourced trades whose callback/atr happened to land
    there. Only atr_floor trades are governed by the multiplier.
    """
    rows = cc.enrich([_trade(source="atr_floor"), _trade(source="ratio"),
                      _trade(source="")])
    assert len(rows) == 1


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


def test_a_missing_journal_is_not_a_crash(capsys, tmp_path):
    sys.argv = ["x", "--journal", str(tmp_path / "nope.jsonl")]
    assert cc.main() == 1
