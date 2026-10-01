"""
tools/scorecard.py — score each MECHANISM against its own goal, not P&L.

Four changes are live at once and P&L cannot separate them. But each was
built to do ONE measurable thing, observable long before any P&L signal.

A mechanism missing its own target is broken whatever P&L does. One hitting
its target has done its job — whether the goal was worth having is a
separate and much slower argument.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import scorecard as sc          # noqa: E402

SCAN = "scan_runner Scan: 12 candidate(s) from 30 movers in 9.0s"


def _run(lines, capsys, tmp_path, trades=""):
    lg = tmp_path / "l.log"
    lg.write_text("\n".join(lines))
    tj = tmp_path / "t.jsonl"
    tj.write_text(trades)
    sys.argv = ["x", str(lg), "--trades", str(tj)]
    sc.main()
    return capsys.readouterr().out


def test_supply_is_scored_against_the_mover_target(capsys, tmp_path):
    out = _run([SCAN, SCAN], capsys, tmp_path)
    assert "movers/scan 30.0" in out
    assert "PASS" in out


def test_supply_MISSES_and_names_the_right_knob(capsys, tmp_path):
    out = _run(["Scan: 3 candidate(s) from 12 movers in 9.0s"] * 2,
               capsys, tmp_path)
    assert "MISS" in out
    assert "SCAN_MIN_CHANGE_PCT" in out, "the volume floor is NOT the knob"


def test_maker_fill_rate_is_measured_on_RESOLVED_not_placed(capsys, tmp_path):
    # Pending orders are not failures; counting them as such understates the
    # rate and would trip the kill criterion early.
    out = _run([SCAN,
                "MAKER ENTRY LONG A", "MAKER ENTRY LONG B", "MAKER ENTRY LONG C",
                "A: entry 1 FILLED — a position is open",
                "B: cancelled stale entry order 9"], capsys, tmp_path)
    assert "fill rate 50%" in out
    assert "pending 1" in out


def test_a_low_fill_rate_mentions_the_HALT_confound(capsys, tmp_path):
    """
    A daily halt cancels resting entry orders, which counts them as
    not-filled through no fault of the mechanism. That contaminated the
    2026-09-26 reading and produced a wrong 'below the kill threshold' call.
    """
    out = _run([SCAN, "MAKER ENTRY A", "MAKER ENTRY B", "MAKER ENTRY C",
                "A: entry 1 FILLED — a position is open",
                "B: cancelled stale entry order 1",
                "C: cancelled stale entry order 2"], capsys, tmp_path)
    assert "MISS" in out and "halt" in out


def test_the_DEFER_vs_SWEEPS_verdict_is_stated(capsys, tmp_path):
    """
    The defer requires CONTINUOUS qualification over ~4 cycles; sweeps=2
    requires it at 2 scan boundaries ~120s apart. The defer is strictly
    harder, so sweeps=2 alongside it is a no-op.
    """
    out = _run([SCAN,
                "auto-trade: A/USDT:USDT DEFERRED — must keep qualifying",
                "auto-trade: A/USDT:USDT defer satisfied — qualified "
                "continuously for 138s, placing the entry"], capsys, tmp_path)
    assert "SWEEPS IS A NO-OP" in out
    assert "median wait 138s" in out


def test_proportional_limits_are_distinguished_from_absolute(capsys, tmp_path):
    out = _run([SCAN,
                "2.10% from the 24h low, limit 0.80% (0.20 x 4.0% move)",
                "4.00% from the 24h high, limit 3.00% (absolute)"],
               capsys, tmp_path)
    assert "proportional 1" in out and "absolute 1" in out


def test_silence_on_distance_is_NOT_read_as_evidence(capsys, tmp_path):
    # The reason string prints only when the gate FIRES.
    out = _run([SCAN], capsys, tmp_path)
    assert "silence is not evidence" in out


def test_a_gate_blocking_over_80pc_is_called_out(capsys, tmp_path):
    """
    BTC_REGIME_MODE would have blocked 91% — moving it to 'block' would have
    near-halted the best trading week of the investigation.
    """
    out = _run([SCAN, "BTC regime (warn) — would BLOCK short. "
                      "193/212 blocked so far."], capsys, tmp_path)
    assert "91%" in out
    assert "near-halting" in out


def test_the_stop_floor_is_read_from_the_JOURNAL_not_the_log(capsys, tmp_path):
    import json, time
    rows = []
    for i in range(6):
        rows.append(json.dumps({
            "opened_at": time.time() - 60, "leverage": 10.0,
            "net_pnl_usdt": 0.1 if i % 2 else -0.2,
            "entry_context": {"auto": True, "sized_stop_roi": 12.0}}))
    out = _run([SCAN], capsys, tmp_path, trades="\n".join(rows))
    assert "median 1.200% of price" in out
    assert "payoff ratio 0.50" in out


def test_a_malformed_journal_line_does_not_crash(capsys, tmp_path):
    # Partial writes have produced non-object lines before.
    out = _run([SCAN], capsys, tmp_path, trades='[]\nnot json\n{"x":1}')
    assert "SCORECARD" in out


def test_a_missing_journal_does_not_crash(capsys, tmp_path):
    lg = tmp_path / "l.log"
    lg.write_text(SCAN)
    sys.argv = ["x", str(lg), "--trades", str(tmp_path / "nope.jsonl")]
    assert sc.main() == 0
