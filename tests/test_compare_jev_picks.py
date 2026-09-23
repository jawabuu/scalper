"""
tools/compare_jev_picks.py — "would trading jev's picks beat trading mine?"

The whole difficulty is that the two populations are NOT comparable: real
trades carry fills, stops, trails, fail-fast and slippage; jev's picks are
idealised forward moves from the decision price. These tests pin the charges
the tool DOES apply and the honesty of what it says about the rest.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_jev_picks as cj          # noqa: E402


def _trade(final_roi=1.0, lev=10.0):
    return {"final_roi": final_roi, "leverage": lev}


def _pick(fav=0.5, adv=0.1, atr=1.0, verdict="ENTER", bot="SKIP"):
    return {"jev_verdict": verdict, "bot_decision": bot, "atr_pct": atr,
            "favoured_side_pct": {"2": fav}, "adverse_pct": {"2": adv}}


def test_both_sides_are_charged_the_SAME_fee():
    """
    Fees are ~0.09% of price per round trip and do not scale with leverage.
    Charging only the real side would hand jev a free 0.09% per trade.
    """
    bot = cj.bot_trades([_trade(final_roi=1.0, lev=10.0)])   # 0.10% gross
    jev, _ = cj.jev_picks([_pick(fav=0.10)], stop_atr=None)
    assert bot[0] == jev[0], "identical gross must give identical net"


def test_everything_is_in_price_percent_not_ROI():
    a = cj.bot_trades([_trade(final_roi=1.0, lev=10.0)])
    b = cj.bot_trades([_trade(final_roi=2.0, lev=20.0)])
    assert a == b


def test_only_picks_the_BOT_SKIPPED_are_counted():
    # A candidate the bot also entered is in the real-trades side already;
    # counting it twice would compare the bot against itself.
    rows = [_pick(bot="SKIP"), _pick(bot="ENTER"), _pick(verdict="SKIP")]
    jev, _ = cj.jev_picks(rows, stop_atr=None)
    assert len(jev) == 1


def test_a_pick_whose_adverse_excursion_breaches_the_stop_is_booked_there():
    """
    The largest chargeable uncharged cost. A pick that ran 3 ATRs against the
    trade before closing favourably did not survive — booking it at its close
    is the single most flattering error available.
    """
    rows = [_pick(fav=+2.0, adv=3.0, atr=1.0)]
    jev, stopped = cj.jev_picks(rows, stop_atr=1.5)
    assert stopped == 1
    assert jev[0] < 0, "a stopped-out pick must book a LOSS, not its close"


def test_the_stop_can_be_disabled():
    rows = [_pick(fav=+2.0, adv=3.0, atr=1.0)]
    jev, stopped = cj.jev_picks(rows, stop_atr=None)
    assert stopped == 0 and jev[0] > 0


def test_a_pick_inside_the_stop_keeps_its_close():
    rows = [_pick(fav=+0.5, adv=0.2, atr=1.0)]
    jev, stopped = cj.jev_picks(rows, stop_atr=1.5)
    assert stopped == 0


def test_the_output_names_what_is_still_uncharged(capsys, tmp_path):
    """
    Entry fill, exit logic, slippage and capacity cannot be charged from this
    data. A tool that prints a gap without naming them invites exactly the
    wrong conclusion.
    """
    t = tmp_path / "t.jsonl"; o = tmp_path / "o.jsonl"
    t.write_text(json.dumps(_trade()) + "\n")
    o.write_text(json.dumps(_pick()) + "\n")
    sys.argv = ["x", "--trades", str(t), "--outcomes", str(o)]
    cj.main()
    out = capsys.readouterr().out
    for phrase in ("ENTRY FILL", "EXIT LOGIC", "SLIPPAGE", "CAPACITY",
                   "NOT evidence"):
        assert phrase.lower() in out.lower(), phrase


def test_a_small_POSITIVE_gap_is_called_UNPROVEN(capsys, tmp_path):
    # 0.10% gross on the bot side vs 0.22% on jev's: a +0.12% gap, inside the
    # uncharged band, so it could vanish once fill and exit logic are charged.
    t = tmp_path / "t.jsonl"; o = tmp_path / "o.jsonl"
    t.write_text("\n".join(json.dumps(_trade(final_roi=1.0)) for _ in range(20)))
    o.write_text("\n".join(json.dumps(_pick(fav=0.22)) for _ in range(20)))
    sys.argv = ["x", "--trades", str(t), "--outcomes", str(o), "--stop-atr", "0"]
    cj.main()
    out = capsys.readouterr().out
    assert "POSITIVE gap smaller than the uncharged costs" in out
    assert "Not evidence" in out


def test_the_uncharged_band_is_ONE_SIDED(capsys, tmp_path):
    """
    Every uncharged cost falls on the jev side, so it can only move the gap
    DOWN. A negative gap of any size is therefore conclusive, while a small
    positive one is not. The first version used a symmetric band and would
    have called the real 2026-09-23 result (-0.123%) "not evidence either
    way".
    """
    t = tmp_path / "t.jsonl"; o = tmp_path / "o.jsonl"
    t.write_text("\n".join(json.dumps(_trade(final_roi=1.0)) for _ in range(20)))
    o.write_text("\n".join(json.dumps(_pick(fav=0.02)) for _ in range(20)))
    sys.argv = ["x", "--trades", str(t), "--outcomes", str(o), "--stop-atr", "0"]
    cj.main()
    out = capsys.readouterr().out
    assert "WORSE while holding every advantage" in out
    assert "CONCLUSIVE" in out
    assert "Not evidence" not in out


def test_missing_logs_are_not_a_crash(tmp_path):
    sys.argv = ["x", "--trades", str(tmp_path / "a"), "--outcomes", str(tmp_path / "b")]
    assert cj.main() == 1
