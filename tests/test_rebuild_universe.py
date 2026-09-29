"""
tools/rebuild_universe.py — reconstruct a past scan, and PROVE it first.

The one finding that matters (shorts in a falling BTC, p=0.0064) rests on
n=79, because BTC fell on only a handful of days. History is the only way to
grow that sample.

But the scanner picks from a 24h TICKER SNAPSHOT that Binance does not serve
historically, so this rebuilds it from klines. That is a reconstruction, not
a recording — hence validation-first. If it cannot reproduce a week we have
logs for, it cannot be trusted on a year we do not.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rebuild_universe as ru          # noqa: E402


class _Ex:
    """Klines shaped like Binance's raw endpoint: index 7 IS quote volume."""

    def __init__(self, table):
        self.table = table
        self.markets = {}

    def fapiPublicGetKlines(self, params):
        sym = params["symbol"]
        rows = self.table.get(sym, [])
        end = params["startTime"] + 200 * 3600 * 1000
        return [r for r in rows if params["startTime"] <= r[0] < end]


def _bars(start_ms, n=100, close0=100.0, drift=0.0, qv=1e6, step_ms=900_000):
    out = []
    c = close0
    for i in range(n):
        o = c
        c = o * (1 + drift)
        out.append([start_ms + i*step_ms, str(o), str(c*1.01), str(o*0.99),
                    str(c), "0", 0, str(qv), 0, "0", "0", "0"])
    return out


def test_quote_volume_comes_from_INDEX_7_not_base_volume():
    """
    ccxt's fetch_ohlcv returns BASE volume and drops quote volume entirely.
    The scanner's percentile floor is computed from QUOTE volume — a
    different number. Using the wrong one moves the floor and changes which
    symbols pass.
    """
    at = 1_790_000_000
    ex = _Ex({"AAAUSDT": _bars(int((at - 25*3600) * 1000), n=120, qv=2e6)})
    qv, pct = ru.window_stats(ex, "AAA/USDT:USDT", at)
    assert qv == 96 * 2e6, qv          # 96 fifteen-minute bars in 24h


def test_a_finer_interval_is_the_DEFAULT():
    """
    Binance's 24h ticker ROLLS; a kline sum is bucketed, and the coarser the
    bucket the worse they agree. With 1h bars the first validation run missed
    12% of the bot's own candidates — and the boundary symbols are exactly
    the ones near the top-N cut.
    """
    import inspect
    sig = inspect.signature(ru.window_stats)
    assert sig.parameters["interval"].default == "15m"


def test_change_is_measured_across_the_FULL_24h_window():
    at = 1_790_000_000
    ex = _Ex({"AAAUSDT": _bars(int((at - 25*3600) * 1000), n=120, drift=0.001)})
    qv, pct = ru.window_stats(ex, "AAA/USDT:USDT", at)
    assert 9 < pct < 11, pct           # 0.1% compounded over 96 bars


def test_an_incomplete_window_returns_None_rather_than_guessing():
    # Fewer than 24 bars means the 24h figure cannot be formed. A partial sum
    # would silently understate volume and mis-rank the symbol.
    at = 1_790_000_000
    ex = _Ex({"AAAUSDT": _bars(int((at - 25*3600) * 1000), n=5)})
    assert ru.window_stats(ex, "AAA/USDT:USDT", at) == (None, None)


def test_recorded_scans_groups_by_scan_bucket(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = []
    for i in range(6):
        rows.append({"ts": 1000 + i, "symbol": "S%d/USDT:USDT" % i})
    for i in range(6):
        rows.append({"ts": 5000 + i, "symbol": "T%d/USDT:USDT" % i})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    got = ru.recorded_scans(p, 10)
    assert len(got) == 2
    assert all(len(s) == 6 for _, s in got)


def test_scans_with_too_few_symbols_are_skipped(tmp_path):
    # A bucket with one symbol cannot score a reconstruction either way.
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"ts": 1000, "symbol": "A/USDT:USDT"}))
    assert ru.recorded_scans(p, 10) == []


def test_the_percentile_floor_is_interpolated_not_indexed():
    # An indexed percentile jumps between samples; the scanner interpolates,
    # and the floor decides which symbols pass at all.
    import inspect
    src = inspect.getsource(ru.rebuild)
    assert "vols[lo] + (vols[hi] - vols[lo]) * (k - lo)" in src


def test_the_docstring_names_every_unfixable_bias():
    # Survivorship, the percentile's dependence on composition, funding, and
    # rolling-vs-bucketed ticker. A backtest tool that hides these invites
    # exactly the over-confidence this month has repeatedly punished.
    d = ru.__doc__.upper()
    for k in ("SURVIVORSHIP", "PERCENTILE", "FUNDING", "TICKER"):
        assert k in d, k


def test_validation_is_the_documented_FIRST_step():
    d = ru.__doc__
    assert "DO THIS FIRST" in d
    assert "cannot be trusted on a year" in d
