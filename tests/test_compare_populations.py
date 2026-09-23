"""
tools/compare_populations.py — "would trading jev's picks have beaten mine?"

Answerable because BOTH populations are in the shadow log: every candidate
carries the bot's decision and jev's verdict, and the resolver labels the
forward path of both. So it is hypothetical-vs-hypothetical, not hypothetical
against the bot's real fills, which carry stops, trails and fees a refused
candidate never sees.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_populations as cp          # noqa: E402


def _row(bot="SKIP", jev="SKIP", fav=0.1, atr=0.8, adv=0.25, edge=1.0):
    return {"bot_decision": bot, "jev_verdict": jev, "atr_pct": atr,
            "favoured_side_pct": {"2": fav}, "adverse_pct": {"2": adv},
            "edge_ratio": {"2": edge}}


def test_the_four_populations_are_disjoint():
    rows = [_row("ENTER", "SKIP"), _row("SKIP", "ENTER"),
            _row("ENTER", "ENTER"), _row("SKIP", "SKIP")]
    g = cp.bucket(rows)
    assert [len(g[k]) for k in ("bot only", "jev only", "both", "neither")] == [1, 1, 1, 1]
    assert sum(len(v) for v in g.values()) == len(rows), "no row may be counted twice"


def test_fees_are_subtracted_in_PRICE_terms():
    """
    ~0.09% of price per round trip, unchanged by leverage. On a population
    whose median move is ~0.25%, that is most of the answer, so it must be
    explicit rather than left for the reader.
    """
    assert cp.FEE_PCT_ROUND_TRIP == 0.09


def test_capacity_is_called_out_when_jev_dwarfs_the_bot(capsys, tmp_path):
    """
    ENTRY_MAX_POSITIONS=6 and an 1800s symbol cooldown. jev's population is
    ~40x the bot's, so most of it could never be taken — a per-trade edge
    there is not an achievable edge.
    """
    p = tmp_path / "o.jsonl"
    rows = [_row("ENTER", "SKIP") for _ in range(10)] + \
           [_row("SKIP", "ENTER") for _ in range(400)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--outcomes", str(p)]
    cp.main()
    out = capsys.readouterr().out
    assert "CAPACITY" in out and "40x" in out
    assert "NOT an achievable edge" in out


def test_a_volatility_difference_between_populations_is_flagged(capsys, tmp_path):
    # The trap that killed three earlier findings: a calmer population shows a
    # smaller adverse excursion mechanically.
    p = tmp_path / "o.jsonl"
    rows = [_row("ENTER", "SKIP", atr=1.2) for _ in range(20)] + \
           [_row("SKIP", "ENTER", atr=0.6) for _ in range(20)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--outcomes", str(p)]
    cp.main()
    out = capsys.readouterr().out
    assert "ATR DIFFERS" in out
    assert "dimensionless" in out


def test_it_says_plainly_that_it_is_not_a_backtest(capsys, tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text("\n".join(json.dumps(_row("ENTER", "SKIP")) for _ in range(10)))
    sys.argv = ["x", "--outcomes", str(p)]
    cp.main()
    out = capsys.readouterr().out
    assert "NOT A BACKTEST" in out and "upper bounds" in out


def test_a_tiny_population_is_skipped_rather_than_reported(capsys, tmp_path):
    # Four observations is not a population; printing a median invites reading
    # one.
    p = tmp_path / "o.jsonl"
    rows = [_row("ENTER", "ENTER") for _ in range(4)] + \
           [_row("SKIP", "SKIP") for _ in range(30)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["x", "--outcomes", str(p)]
    cp.main()
    body = capsys.readouterr().out.split("Horizon")[1]
    assert "both" not in body


def test_a_missing_outcomes_file_is_not_a_crash(capsys, tmp_path):
    sys.argv = ["x", "--outcomes", str(tmp_path / "nope.jsonl")]
    assert cp.main() == 1
