"""
tools/entry_funnel.py — WHICH change cut the entries, and by how much?

Three landed within days and all reduce entries (defer, maker fill rate, the
RSI band), while movers per scan independently fell from ~40 to ~20. Counting
entries cannot separate them; the funnel can, because each stage has its own
log line and its own attrition.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import entry_funnel as ef          # noqa: E402


def _run(lines, capsys, tmp_path):
    p = tmp_path / "l.log"
    p.write_text("\n".join(lines))
    sys.argv = ["x", str(p)]
    ef.main()
    return capsys.readouterr().out


SCAN = "INFO scan_runner Scan: 10 candidate(s) from 20 movers in 9.0s"


def test_the_market_is_reported_separately_from_the_config(capsys, tmp_path):
    """
    Movers per scan halved over the same period the changes landed. That part
    is the market and no setting is responsible for it — reporting it beside
    the config stages is what stops it being misattributed.
    """
    out = _run([SCAN, SCAN], capsys, tmp_path)
    assert "THE MARKET" in out and "movers per scan          20.0" in out


def test_defer_attrition_is_measured(capsys, tmp_path):
    out = _run([
        SCAN,
        "auto-trade: A/USDT:USDT DEFERRED — must keep qualifying for 120s",
        "auto-trade: B/USDT:USDT DEFERRED — must keep qualifying for 120s",
        "auto-trade: A/USDT:USDT defer satisfied — qualified continuously",
    ], capsys, tmp_path)
    assert "started the clock      2" in out
    assert "50% lost here" in out


def test_maker_fill_attrition_is_measured(capsys, tmp_path):
    out = _run([
        SCAN,
        "WARNING futures_entry MAKER ENTRY LONG A: post-only LIMIT",
        "WARNING futures_entry MAKER ENTRY LONG B: post-only LIMIT",
        "ERROR futures_entry A: entry 1 FILLED — a position is open.",
        "WARNING futures_entry B: cancelled stale entry order 9",
    ], capsys, tmp_path)
    assert "fill rate              50% of resolved" in out


def test_refusals_are_shown_as_SHARES_not_counts(capsys, tmp_path):
    """
    The refusal counter is per CYCLE and cycles repeat the same candidates, so
    absolute numbers mean nothing. Only the share between reasons is readable.
    """
    out = _run([SCAN,
                "auto-trade refusals this cycle: rsi_band=5, min_atr=5"],
               capsys, tmp_path)
    assert "rsi_band                   50%" in out
    assert "share, not count" in out


def test_an_ACTIVE_HALT_is_called_out(capsys, tmp_path):
    # Otherwise a window containing a halt looks like a config problem.
    out = _run([SCAN, "AUTO-TRADE STILL HALTED (2.1h): daily loss limit"],
               capsys, tmp_path)
    assert "daily halt was ACTIVE" in out


def test_an_inactive_change_says_so_rather_than_printing_zero(capsys, tmp_path):
    out = _run([SCAN], capsys, tmp_path)
    assert "no DEFERRED lines" in out
    assert "no MAKER ENTRY lines" in out


def test_a_log_with_no_scans_is_an_error(capsys, tmp_path):
    p = tmp_path / "l.log"
    p.write_text("nothing useful here")
    sys.argv = ["x", str(p)]
    assert ef.main() == 1
