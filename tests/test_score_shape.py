"""tools/score_shape.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import score_shape as ss          # noqa: E402



def _t(net, **ctx):
    return {"net_pnl_usdt": net, "entry_context": {"shape_ok": True, **ctx}}


def test_ctx_reads_the_PREFIXED_name():
    """
    v3.86.1 wrote the path shape as shape_compression etc; this tool read the
    BARE name. Result: "193 trades, 34 with shape recorded" and an EMPTY rule
    table. The same name-collision bug twice in one day, once in each
    direction.
    """
    t = _t(1.0, shape_compression=0.5)
    assert ss._ctx(t, "compression") == 0.5
    assert ss._ctx(t, "shape_compression") == 0.5


def test_ctx_still_reads_a_BARE_name_from_an_older_row():
    # A mixed journal must score whole, not silently drop half its rows.
    t = _t(1.0, compression=0.5)
    assert ss._ctx(t, "compression") == 0.5
    assert ss._ctx(t, "shape_compression") == 0.5


def test_a_rule_that_cannot_split_SAYS_SO(capsys, tmp_path):
    """
    A rule keeping everything printed nothing, which reads as a broken tool.
    That silence is exactly what hid the mismatch for two runs.
    """
    j = tmp_path / "t.jsonl"
    j.write_text("\n".join(
        json.dumps(_t(1.0, shape_extension_atr=0.1)) for _ in range(25)))
    ss.main(str(j))
    out = capsys.readouterr().out
    assert "no split" in out


def test_a_thin_sample_is_called_out(capsys, tmp_path):
    j = tmp_path / "t.jsonl"
    j.write_text("\n".join(
        json.dumps(_t(1.0, shape_compression=i / 10)) for i in range(5)))
    ss.main(str(j))
    assert "smoke test, not a result" in capsys.readouterr().out
