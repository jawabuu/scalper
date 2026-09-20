"""
Forward-return labelling (bot/shadow_outcomes.py).

The point of these: the resolver exists to make refused candidates
measurable, and the two ways it could quietly lie are by inventing prices
where candles are missing and by treating repeated judgements of one coin as
independent observations. Both are pinned here.
"""

import json
import time

import pytest

from bot.shadow_outcomes import (
    collapse, resolve, summarize, ShadowOutcome, HORIZONS_MIN,
)


class _Exchange:
    """Minimal ccxt-shaped fake. 1m candles at a fixed drift."""

    def __init__(self, start=100.0, drift=0.0, minutes=180, gap_after=None):
        self.start, self.drift, self.minutes = start, drift, minutes
        self.gap_after = gap_after       # stop emitting after N minutes
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append((symbol, timeframe, since, limit))
        out = []
        n = self.minutes if self.gap_after is None else self.gap_after
        for i in range(min(n, limit)):
            t = since + i * 60_000
            px = self.start + self.drift * i
            out.append([t, px, px, px, px, 1000.0])
        return out


def _row(symbol="X/USDT:USDT", ts=None, side="short", bot="SKIP",
         verdict="SKIP", **kw):
    base = {
        "ts": ts if ts is not None else time.time() - 7200,
        "symbol": symbol, "side": side, "bot_decision": bot,
        "jev_verdict": verdict, "confidence": 0.4, "conviction": 1.2,
        "looks_exhausted": 0.47, "regime_aligned": 0.11,
        "structure_intact": 0.4, "composed_score": 0.34,
    }
    base.update(kw)
    return base


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")


# ── Collapsing — the independence problem ───────────────────────────────────

def test_repeated_judgements_of_one_coin_are_one_observation():
    # The sample that motivated this: 1021 rows, 24 symbols, CHILLGUY 106
    # times. Labelling each separately would manufacture confidence.
    t = 1_000_000.0
    rows = [_row(symbol="CHILLGUY/USDT:USDT", ts=t + i * 10) for i in range(106)]
    groups = collapse(rows)
    assert len(groups) == 1
    assert len(groups[0]) == 106


def test_the_same_coin_much_later_is_a_separate_observation():
    t = 1_000_000.0
    rows = [_row(ts=t), _row(ts=t + 5000)]
    assert len(collapse(rows)) == 2


def test_different_symbols_are_never_collapsed_together():
    t = 1_000_000.0
    rows = [_row(symbol="A/USDT:USDT", ts=t), _row(symbol="B/USDT:USDT", ts=t)]
    assert len(collapse(rows)) == 2


def test_the_two_sides_of_one_symbol_stay_separate():
    t = 1_000_000.0
    rows = [_row(ts=t, side="short"), _row(ts=t + 5, side="long")]
    assert len(collapse(rows)) == 2


def test_unknown_rows_are_dropped_not_labelled():
    # Everything before v3.69.0 is UNKNOWN — a failed call, not a judgement.
    rows = [_row(verdict="UNKNOWN"), _row(verdict="UNKNOWN")]
    assert collapse(rows) == []


def test_the_earliest_row_of_a_group_is_the_one_kept():
    t = 1_000_000.0
    rows = [_row(ts=t + 30, conviction=9.9), _row(ts=t, conviction=1.1)]
    groups = collapse(rows)
    assert groups[0][0]["ts"] == t
    assert groups[0][0]["conviction"] == 1.1


# ── Labelling ───────────────────────────────────────────────────────────────

def test_a_ripe_decision_is_labelled_with_forward_returns(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    ex = _Exchange(start=100.0, drift=-0.1)      # falling
    assert resolve(ex, str(src), str(dst), now=now) == 1
    rec = json.loads(dst.read_text().strip())
    assert set(rec["returns_pct"]) == {"15", "30", "60"}
    assert rec["returns_pct"]["60"] < 0


def test_a_short_that_fell_reads_as_FAVOURED_not_negative(tmp_path):
    # The sign trap: without this a winning short averages negative and every
    # later statistic silently inverts.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["returns_pct"]["60"] < 0
    assert rec["favoured_side_pct"]["60"] > 0


def test_a_long_keeps_its_sign(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="long")])
    resolve(_Exchange(start=100.0, drift=0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["favoured_side_pct"]["60"] > 0
    assert rec["favoured_side_pct"]["60"] == rec["returns_pct"]["60"]


def test_an_unripe_decision_is_left_for_a_later_run(tmp_path):
    # Writing a truncated observation would be worse than writing none.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 300)])            # 5 min old, 60 min horizon
    assert resolve(_Exchange(), str(src), str(dst), now=now) == 0
    assert not dst.exists()


def test_a_missing_minute_is_missing_not_interpolated(tmp_path):
    # Candles stop at 20 minutes: 15 resolves, 30 and 60 must be absent
    # rather than filled from the last known price.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    resolve(_Exchange(drift=-0.1, gap_after=20), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert "15" in rec["returns_pct"]
    assert "30" not in rec["returns_pct"]
    assert "60" not in rec["returns_pct"]


def test_no_candles_at_all_writes_nothing(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    assert resolve(_Exchange(gap_after=0), str(src), str(dst), now=now) == 0


def test_a_fetch_failure_skips_that_row_and_continues(tmp_path):
    class _Flaky(_Exchange):
        def fetch_ohlcv(self, symbol, *a, **kw):
            if symbol.startswith("BAD"):
                raise RuntimeError("network")
            return super().fetch_ohlcv(symbol, *a, **kw)

    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(symbol="BAD/USDT:USDT", ts=now - 7200),
                 _row(symbol="OK/USDT:USDT", ts=now - 7200)])
    assert resolve(_Flaky(drift=-0.1), str(src), str(dst), now=now) == 1


def test_the_collapsed_count_is_recorded_so_sample_size_is_visible(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    t = now - 7200
    _write(src, [_row(ts=t + i * 10) for i in range(40)])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["collapsed_rows"] == 40, \
        "a reader must see 1 observation covering 40 rows, not 40 observations"


def test_re_running_does_not_duplicate(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    ex = _Exchange(drift=-0.1)
    assert resolve(ex, str(src), str(dst), now=now) == 1
    assert resolve(ex, str(src), str(dst), now=now) == 0
    assert len(dst.read_text().strip().splitlines()) == 1


def test_the_decision_log_is_never_rewritten(tmp_path):
    # Append-only, written live. Adding a column in place invites a torn write.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    before = src.read_text()
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    assert src.read_text() == before


def test_a_missing_decision_log_is_not_an_error(tmp_path):
    assert resolve(_Exchange(), str(tmp_path / "nope.jsonl"),
                   str(tmp_path / "o.jsonl"), now=1_000_000.0) == 0


def test_an_unparseable_line_does_not_stop_the_pass(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    src.write_text(json.dumps(_row(ts=now - 7200)) + "\n{ broken\n",
                   encoding="utf-8")
    assert resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now) == 1


# ── Reading ─────────────────────────────────────────────────────────────────

def test_summarize_reports_observations_AND_rows_covered(tmp_path):
    # The distinction that stops 1021 rows reading as 1021 data points.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(symbol="A/USDT:USDT", ts=now - 7200 + i * 10)
                 for i in range(30)] +
                [_row(symbol="B/USDT:USDT", ts=now - 7200)])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert s["observations"] == 2
    assert s["decision_rows_covered"] == 31


def test_summarize_splits_by_verdict(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(symbol="A/USDT:USDT", ts=now - 7200, verdict="SKIP"),
                 _row(symbol="B/USDT:USDT", ts=now - 7200, verdict="ENTER")])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert set(s["by_verdict"]) == {"SKIP", "ENTER"}
    assert s["by_verdict"]["SKIP"]["n"] == 1


def test_summarize_on_an_empty_file_is_not_an_error(tmp_path):
    s = summarize(str(tmp_path / "none.jsonl"))
    assert s["observations"] == 0
