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

    def __init__(self, start=100.0, drift=0.0, minutes=180, gap_after=None,
                 wick=0.0):
        self.start, self.drift, self.minutes = start, drift, minutes
        self.gap_after = gap_after       # stop emitting after N minutes
        self.wick = wick                 # symmetric high/low around the close
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append((symbol, timeframe, since, limit))
        out = []
        n = self.minutes if self.gap_after is None else self.gap_after
        for i in range(min(n, limit)):
            t = since + i * 60_000
            px = self.start + self.drift * i
            hi = px + (self.wick or 0.0)
            lo = px - (self.wick or 0.0)
            out.append([t, px, hi, lo, px, 1000.0])
        return out


def _row(symbol="X/USDT:USDT", ts=None, side="short", bot="SKIP",
         verdict="SKIP", atr=1.0, **kw):
    base = {
        "inputs_seen": {"candidate": {"atr_pct": atr}},
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
    assert set(rec["returns_pct"]) == {"1", "2", "3", "5", "10", "30"}
    assert rec["returns_pct"]["30"] < 0


def test_a_short_that_fell_reads_as_FAVOURED_not_negative(tmp_path):
    # The sign trap: without this a winning short averages negative and every
    # later statistic silently inverts.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["returns_pct"]["30"] < 0
    assert rec["favoured_side_pct"]["30"] > 0


def test_a_long_keeps_its_sign(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="long")])
    resolve(_Exchange(start=100.0, drift=0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["favoured_side_pct"]["30"] > 0
    assert rec["favoured_side_pct"]["30"] == rec["returns_pct"]["30"]


def test_an_unripe_decision_is_left_for_a_later_run(tmp_path):
    # Writing a truncated observation would be worse than writing none.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 60)])             # 1 min old, 30 min horizon
    assert resolve(_Exchange(), str(src), str(dst), now=now) == 0
    assert not dst.exists()


def test_a_missing_minute_is_missing_not_interpolated(tmp_path):
    # Candles stop at 20 minutes: 15 resolves, 30 and 60 must be absent
    # rather than filled from the last known price.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    resolve(_Exchange(drift=-0.1, gap_after=12), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert "10" in rec["returns_pct"]
    assert "30" not in rec["returns_pct"]


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


# ── Competing entry triggers ────────────────────────────────────────────────
#
# Selection is not the problem — 79.1% of entries were already losing when
# first seen, and no entry-context field separated the winners. These pin the
# trigger comparison: CRT, jev's own read, and the plain one-candle wait, all
# scored against the same forward returns.

def test_the_recorded_trigger_survives_onto_the_outcome_row(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200,
                      triggers={"crt_agrees": True, "crt_side": "short"})])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["triggers"]["crt_agrees"] is True


def test_crt_no_opinion_is_scored_apart_from_crt_disagreeing(tmp_path):
    """
    None means CRT abstained; False means it disagreed. Collapsing them would
    score a trigger that had no view as one that was wrong — and on live data
    every entry so far is False or None, so this distinction IS the dataset.
    """
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [
        _row(symbol="A/USDT:USDT", ts=now - 7200, triggers={"crt_agrees": False}),
        _row(symbol="B/USDT:USDT", ts=now - 7200, triggers={"crt_agrees": None}),
        _row(symbol="C/USDT:USDT", ts=now - 7200, triggers={"crt_agrees": True}),
    ])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert set(s["by_crt_agrees"]) == {"True", "False", "None"}
    assert all(v["n"] == 1 for v in s["by_crt_agrees"].values())


def test_room_ahead_is_banded_in_ATR_units(tmp_path):
    """
    The claim: "skip when the obstacle is closer than the stop." Banded in ATR
    units so it survives every leverage and config change in this
    investigation.

    Stated prediction: if the claim holds here, <1 ATR should be worse. If
    momentum position is what matters instead, the bands will be flat — a
    level 2% away is no obstacle to a 1.96-minute trade.
    """
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [
        _row(symbol="A/USDT:USDT", ts=now - 7200, triggers={"room_ahead_atr": 0.4}),
        _row(symbol="B/USDT:USDT", ts=now - 7200, triggers={"room_ahead_atr": 1.5}),
        _row(symbol="C/USDT:USDT", ts=now - 7200, triggers={"room_ahead_atr": 6.0}),
    ])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert list(s["by_room_ahead_atr"]) == ["<1 ATR", "1-2", ">4"], \
        "bands must stay in ascending order, not dict insertion order"


def test_a_missing_room_value_is_omitted_not_bucketed(tmp_path):
    # Absent must not silently land in the lowest band.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, triggers={"crt_agrees": True})])
    resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert not s.get("by_room_ahead_atr")


def test_a_row_with_no_triggers_still_resolves(tmp_path):
    # Rows written before this existed carry none, and must not be dropped.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200)])
    assert resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now) == 1
    assert json.loads(dst.read_text().strip())["triggers"] == {}


def test_the_one_minute_wait_baseline_is_reported(tmp_path):
    """
    The comparison that keeps a structural trigger honest: if CRT cannot beat
    simply waiting one candle, its structure has earned nothing.
    """
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    one = s["wait_baseline"]["one_minute"]
    assert one["n"] == 1
    # A falling market favours the short at every horizon.
    assert one["median_favoured_pct"] > 0
    assert one["better_than_now_pct"] == 100.0


# ── Path, not endpoint ──────────────────────────────────────────────────────
#
# The operator's criterion is "direction right AND the adverse excursion
# small — consolidating before the move". Closes cannot express that: a
# candidate that ran -8% before +12% scores identically to one that went
# straight to +12%.

def test_a_short_that_only_fell_has_ZERO_adverse_excursion(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["adverse_pct"]["30"] == 0.0
    assert rec["favourable_pct"]["30"] > 0


def test_adverse_is_never_NEGATIVE(tmp_path):
    # A clamp, not a cosmetic one: a negative "adverse" would flatter every
    # later average of this column.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="long")])
    resolve(_Exchange(start=100.0, drift=0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["adverse_pct"]["30"] >= 0.0


def test_excursions_use_WICKS_not_closes(tmp_path):
    # What the position would have lived through. A close hides the wick that
    # would have taken out a stop.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1, wick=2.0), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["adverse_pct"]["30"] > 1.0, \
        "an upper wick is adverse for a short even if every close fell"


def test_the_edge_ratio_is_NONE_not_infinity_when_nothing_went_against(tmp_path):
    # inf poisons every median it lands in, and an undefined ratio is not a
    # large edge.
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(ts=now - 7200, side="short")])
    resolve(_Exchange(start=100.0, drift=-0.1), str(src), str(dst), now=now)
    rec = json.loads(dst.read_text().strip())
    assert rec["edge_ratio"]["30"] is None


def test_summarize_reports_path_quality_split_by_trigger(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [
        _row(symbol="A/USDT:USDT", ts=now - 7200, triggers={"crt_agrees": True}),
        _row(symbol="B/USDT:USDT", ts=now - 7200, triggers={"crt_agrees": False}),
    ])
    resolve(_Exchange(drift=-0.1, wick=0.5), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=30)
    assert s["path"]["n"] == 2
    assert s["path"]["median_adverse_pct"] is not None
    assert set(s["path_by_crt_agrees"]) == {"True", "False"}
    assert "adverse_under_0.5pct" in s["path"]


def test_the_horizons_match_the_strategys_actual_hold_time():
    """
    Measured over 771 trades: median hold 1.3 min, 80% closed within 3 min,
    99% within 30. The first version scored every trigger at 30 minutes — a
    window 23x longer than the median trade, i.e. it measured whether a
    trigger predicts something the bot is never exposed to.

    1 minute is the floor (no finer candle from fetch_ohlcv). If these ever
    drift long again, trigger conclusions become meaningless.
    """
    from bot.shadow_outcomes import HORIZONS_MIN
    assert min(HORIZONS_MIN) == 1
    assert sorted(HORIZONS_MIN)[:3] == [1, 2, 3], \
        "the median trade lives inside the first 3 minutes"


def test_summarize_defaults_to_a_SCALPER_horizon():
    import inspect
    from bot.shadow_outcomes import summarize
    d = inspect.signature(summarize).parameters["horizon"].default
    assert d <= 3, f"default horizon {d} is longer than 80% of trades live"


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


def test_resolve_reports_progress_before_it_starts_fetching(tmp_path, caplog):
    """
    One candle fetch per observation, rate-limited, through a proxy. A rebuild
    of a few hundred rows runs for minutes, and the only output used to be at
    the very end — which reads as a hang.
    """
    import logging
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(symbol=f"S{i}/USDT:USDT", ts=now - 7200) for i in range(60)])
    with caplog.at_level(logging.INFO):
        resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    text = caplog.text
    assert "resolving 60 observation(s)" in text
    assert "25/60" in text, "long runs must report progress, not just a total"


def test_progress_is_silent_for_a_small_run(tmp_path, caplog):
    # Noise on a three-row run helps nobody.
    import logging
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [_row(symbol=f"S{i}/USDT:USDT", ts=now - 7200) for i in range(3)])
    with caplog.at_level(logging.INFO):
        resolve(_Exchange(drift=-0.1), str(src), str(dst), now=now)
    assert "/3 ..." not in caplog.text


def test_each_group_reports_its_MEDIAN_ATR(tmp_path):
    """
    A group with lower ATR shows a smaller adverse excursion MECHANICALLY.
    Without ATR beside it, jev's ENTER cohort looking calmer than SKIP cannot
    be told apart from jev simply picking calmer candidates — the same
    endogenous-split trap as the 'ratio' cohort in compare_callback_mult.py.
    """
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    _write(src, [
        _row(symbol="A/USDT:USDT", ts=now - 7200, verdict="ENTER", atr=0.4),
        _row(symbol="B/USDT:USDT", ts=now - 7200, verdict="SKIP", atr=2.0),
    ])
    resolve(_Exchange(drift=-0.1, wick=0.5), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=2)
    assert s["path_by_verdict"]["ENTER"]["median_atr_pct"] == 0.4
    assert s["path_by_verdict"]["SKIP"]["median_atr_pct"] == 2.0


def test_a_missing_ATR_does_not_break_the_group(tmp_path):
    src, dst = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    now = 1_000_000.0
    r = _row(ts=now - 7200)
    r["inputs_seen"] = {}
    _write(src, [r])
    resolve(_Exchange(drift=-0.1, wick=0.5), str(src), str(dst), now=now)
    s = summarize(str(dst), horizon=2)
    assert s["path"]["median_atr_pct"] is None


def test_the_CLI_horizon_default_matches_the_hold_time():
    """
    summarize() was moved to 2 minutes in v3.75.0 and the CLI default was
    missed, so `--summary` kept reporting the 30-minute window that made the
    first trigger read meaningless. Median hold is 1.96 min.
    """
    import subprocess, sys as _s, pathlib
    tool = pathlib.Path(__file__).resolve().parents[1] / "tools" / "resolve_shadow_outcomes.py"
    out = subprocess.run([_s.executable, str(tool), "--help"],
                         capture_output=True, text=True).stdout
    assert "default: 2" in out or "(default: 2)" in out or "--horizon HORIZON" in out
    src = tool.read_text()
    assert 'p.add_argument("--horizon", type=int, default=2' in src


def test_summarize_on_an_empty_file_is_not_an_error(tmp_path):
    s = summarize(str(tmp_path / "none.jsonl"))
    assert s["observations"] == 0
