"""
Shadow decision log (bot/shadow_decision.py) — built against JEV-BRIEF.md.

Three groups: the writer (fail-open, rate-capped, fire-and-forget, never
touches the real decision), the reader (pure file I/O + join, never calls the
model), and the wiring into auto_trader.py/api.py (source inspection, same
convention the rest of this suite uses for those two files — there is no
TestClient anywhere in this suite; auth is a real GitHub OAuth cookie).
"""
import inspect
import json
import threading
import time

import pytest

from bot.shadow_decision import (
    ShadowDecisionLogger, ShadowDecision, read_decisions, summarize,
    FINGERPRINT, _QUESTION_SPECS, _candidate_state, _reasons,
    VALID_VERDICTS,
)


# ── Fakes ────────────────────────────────────────────────────────────────────

class _Ans:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Result:
    """
    There is no "verdict" answer any more. jev is asked four ATOMIC questions
    and the verdict is composed in code by _compose_verdict, so the weighting
    can be re-fitted offline instead of living in a prompt.

    `verdict` here is a convenience for the tests: it picks component values
    that compose to the wanted side.
    """
    def __init__(self, verdict="enter", confidence=0.8, conviction=None,
                 looks_exhausted=None, regime_aligned=None,
                 structure_intact=None):
        want_enter = str(verdict).lower() != "skip"
        if conviction is None:
            conviction = 2.6 if want_enter else 1.0
        if looks_exhausted is None:
            looks_exhausted = 0.1 if want_enter else 0.9
        if regime_aligned is None:
            regime_aligned = 0.9 if want_enter else 0.15
        if structure_intact is None:
            structure_intact = 0.9 if want_enter else 0.2
        self.model = "jev-test"
        self.choices = {}
        self.scores = {"conviction": _Ans(score=conviction,
                                          confidence=confidence)}
        self.nouls = {
            "looks_exhausted": _Ans(noul=looks_exhausted,
                                    confidence=confidence),
            "regime_aligned": _Ans(noul=regime_aligned, confidence=confidence),
            "structure_intact": _Ans(noul=structure_intact,
                                     confidence=confidence)}


class _Client:
    def __init__(self, result=None, raises=None, delay=0.0):
        self.calls = []
        self._result = result or _Result()
        self._raises = raises
        self._delay = delay
        self._lock = threading.Lock()

    def system_one(self, state, questions):
        with self._lock:
            self.calls.append((state, questions))
        if self._delay:
            time.sleep(self._delay)
        if self._raises:
            raise self._raises
        return self._result


def _row(**kw):
    base = {
        "symbol": "ARB/USDT:USDT", "direction": "short", "rsi": 71.0,
        "atr_pct": 1.2, "ema_gap_pct": -0.4, "change_24h_pct": 9.0,
        "range_pos_24h": 0.9, "advance": {}, "taper": {}, "turn": {},
        "breakout": {}, "efficiency": {},
    }
    base.update(kw)
    return base


def _wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ── State construction — no look-ahead, honest about the gap ────────────────

def test_state_is_built_only_from_the_row_and_snapshot_given():
    # No fetch of any kind happens here — the function is pure.
    state = _candidate_state("ARB/USDT:USDT", "short", _row(), {"breadth_pct": 40})
    assert state["candidate"]["symbol"] == "ARB/USDT:USDT"
    assert state["candidate"]["rsi"] == 71.0
    assert state["regime"]["breadth_pct"] == 40


def test_the_order_book_gap_is_named_explicitly_not_omitted():
    state = _candidate_state("ARB/USDT:USDT", "short", _row(), {})
    assert "UNAVAILABLE" in state["order_book"]


def test_every_question_is_asked_in_one_request():
    assert set(_QUESTION_SPECS) == {"conviction", "looks_exhausted",
                                    "regime_aligned", "structure_intact"}
    assert "verdict" not in _QUESTION_SPECS, \
        "the verdict must be composed in code, not asked of the model"


def test_every_question_judges_exactly_one_thing():
    """
    A composite "would you take this trade?" hands the weighting to the model,
    and a change of priorities then becomes a prompt rewrite whose effect on
    past rows is unknowable.
    """
    from bot.shadow_decision import _WEIGHTS, _ENTER_THRESHOLD
    assert set(_WEIGHTS) == {"not_exhausted", "regime_aligned",
                             "structure_intact", "conviction"}
    assert 0.0 < _ENTER_THRESHOLD < 1.0

def test_a_successful_call_writes_one_json_line(tmp_path):
    path = tmp_path / "shadow.jsonl"
    c = _Client(_Result(verdict="skip", conviction=2.5, looks_exhausted=0.8))
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("ARB/USDT:USDT", "short", _row(),
                        bot_decision="ENTER", snap={"breadth_pct": 30})
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[0])
    assert line["symbol"] == "ARB/USDT:USDT"
    assert line["bot_decision"] == "ENTER"
    assert line["jev_verdict"] == "SKIP"
    assert line["fingerprint"] == FINGERPRINT
    assert "order_book" in line["inputs_seen"]


def test_returns_immediately_even_when_the_call_is_slow(tmp_path):
    c = _Client(delay=1.0)
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    started = time.time()
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="ENTER")
    assert time.time() - started < 0.3


def test_no_api_key_disables_it_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    path = tmp_path / "shadow.jsonl"
    logger = ShadowDecisionLogger(path=str(path))
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    time.sleep(0.2)
    assert not path.exists()


def test_a_client_that_raises_still_logs_an_unknown_row(tmp_path):
    # Constraint 4 (JEV-BRIEF.md §5): UNKNOWN is a valid, RECORDED outcome —
    # a failed call must not just vanish silently.
    path = tmp_path / "shadow.jsonl"
    c = _Client(raises=RuntimeError("502"))
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[0])
    assert line["jev_verdict"] == "UNKNOWN"
    assert "502" in line["reasons"][0]


def test_a_missing_component_is_logged_as_unknown_not_guessed(tmp_path):
    """
    There is no verdict STRING to mis-parse any more. The equivalent risk is a
    component the model did not return: composing a verdict from three of four
    answers would silently change the weighting the row claims to use.
    """
    class _Partial(_Result):
        def __init__(self):
            super().__init__()
            del self.nouls["structure_intact"]

    path = tmp_path / "s.jsonl"
    logger = ShadowDecisionLogger(path=str(path), client=_Client(_Partial()))
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="ENTER")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[-1])
    assert line["jev_verdict"] == "UNKNOWN"
    assert line["bot_decision"] == "ENTER"


def test_the_weights_and_threshold_are_recorded_on_every_row(tmp_path):
    """
    Recorded so the composition can be RE-FITTED offline against outcomes
    without re-running a single call — and so a row composed under different
    weights is never silently compared against one that was not.
    """
    path = tmp_path / "s.jsonl"
    logger = ShadowDecisionLogger(path=str(path),
                                  client=_Client(_Result(verdict="enter")))
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="ENTER")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[-1])
    assert line["weights"] and line["enter_threshold"] > 0
    for k in ("conviction", "looks_exhausted", "regime_aligned",
              "structure_intact", "composed_score"):
        assert k in line, f"{k} must be recorded raw"


def test_confidence_is_the_weakest_component_not_an_average(tmp_path):
    """A verdict from four judgements is only as good as its weakest input."""
    path = tmp_path / "s.jsonl"
    r = _Result(verdict="enter")
    r.nouls["regime_aligned"] = _Ans(noul=0.9, confidence=0.2)
    logger = ShadowDecisionLogger(path=str(path), client=_Client(r))
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="ENTER")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[-1])
    assert line["confidence"] == 0.2


def test_an_invalid_bot_decision_is_rejected_before_any_call(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="MAYBE")
    time.sleep(0.1)
    assert c.calls == []


def test_the_entry_order_id_is_carried_through_to_the_logged_row(tmp_path):
    path = tmp_path / "shadow.jsonl"
    c = _Client()
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="ENTER",
                        entry_order_id="9988776655")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[0])
    assert line["entry_order_id"] == "9988776655"


def test_a_skip_has_no_entry_order_id_by_default(tmp_path):
    path = tmp_path / "shadow.jsonl"
    c = _Client()
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[0])
    assert line["entry_order_id"] is None


# ── The rate cap ─────────────────────────────────────────────────────────────

def test_calls_beyond_the_per_minute_cap_are_dropped_not_queued(tmp_path):
    path = tmp_path / "shadow.jsonl"
    c = _Client()
    logger = ShadowDecisionLogger(path=str(path), client=c, max_per_minute=2)
    for i in range(5):
        logger.decide_async(f"S{i}/USDT:USDT", "long", _row(),
                            bot_decision="SKIP")
    time.sleep(0.3)
    assert len(c.calls) == 2
    assert logger.dropped_for_rate == 3


def test_the_rate_cap_does_not_affect_calls_a_minute_later(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  max_per_minute=1)
    logger.decide_async("A/USDT:USDT", "long", _row(), bot_decision="SKIP")
    time.sleep(0.1)
    logger._recent = [time.time() - 61]   # simulate a minute having passed
    logger.decide_async("B/USDT:USDT", "long", _row(), bot_decision="SKIP")
    time.sleep(0.1)
    assert len(c.calls) == 2


# ── Reading — pure file I/O, never calls the model ───────────────────────────

def test_read_decisions_is_newest_first(tmp_path):
    path = tmp_path / "shadow.jsonl"
    rows = [{"ts": t, "symbol": "A/USDT:USDT", "jev_verdict": "ENTER"}
            for t in (1.0, 3.0, 2.0)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = read_decisions(str(path))
    assert [d["ts"] for d in got] == [3.0, 2.0, 1.0]


def test_read_decisions_filters_by_symbol_verdict_and_since(tmp_path):
    path = tmp_path / "shadow.jsonl"
    rows = [
        {"ts": 10, "symbol": "A/USDT:USDT", "jev_verdict": "ENTER"},
        {"ts": 20, "symbol": "B/USDT:USDT", "jev_verdict": "SKIP"},
        {"ts": 30, "symbol": "A/USDT:USDT", "jev_verdict": "SKIP"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [d["ts"] for d in read_decisions(str(path), symbol="A/USDT:USDT")] == [30, 10]
    assert [d["ts"] for d in read_decisions(str(path), verdict="skip")] == [30, 20]
    assert [d["ts"] for d in read_decisions(str(path), since=15)] == [30, 20]


def test_read_decisions_respects_limit(tmp_path):
    path = tmp_path / "shadow.jsonl"
    rows = [{"ts": t, "symbol": "A"} for t in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert len(read_decisions(str(path), limit=3)) == 3


def test_a_missing_log_file_returns_an_empty_list(tmp_path):
    assert read_decisions(str(tmp_path / "nope.jsonl")) == []


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "shadow.jsonl"
    good = json.dumps({"ts": 1, "symbol": "A"})
    path.write_text(good + "\n{not json\n" + good + "\n")
    assert len(read_decisions(str(path))) == 2


# ── Summary / join ───────────────────────────────────────────────────────────

def _decision(ts, symbol, bot_decision, jev_verdict, entry_order_id=None):
    return {"ts": ts, "symbol": symbol, "side": "long",
            "bot_decision": bot_decision, "jev_verdict": jev_verdict,
            "confidence": 0.8, "conviction": 2.0, "looks_exhausted": 0.3,
            "regime_aligned": 0.7, "reasons": [], "entry_order_id": entry_order_id,
            "model": "jev", "fingerprint": FINGERPRINT, "inputs_seen": {}}


def _trade(order_id, net_pnl):
    return {"symbol": "A/USDT:USDT",
            "entry_context": {"entry_order_id": order_id},
            "net_pnl_usdt": net_pnl}


def test_agreement_counts_only_resolved_verdicts():
    decisions = [
        _decision(1, "A", "ENTER", "ENTER"),   # agrees
        _decision(2, "A", "SKIP", "SKIP"),     # agrees
        _decision(3, "A", "ENTER", "SKIP"),    # disagrees
        _decision(4, "A", "ENTER", "UNKNOWN"), # excluded from the ratio
    ]
    s = summarize(decisions, [])
    assert s["agreement_with_bot"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["n_trials"] == 4


def test_net_pnl_is_split_by_jev_verdict_via_the_order_id_join():
    decisions = [
        _decision(1, "A", "ENTER", "ENTER", entry_order_id="111"),
        _decision(2, "A", "ENTER", "SKIP", entry_order_id="222"),
    ]
    trades = [_trade("111", 5.0), _trade("222", -3.0)]
    s = summarize(decisions, trades)
    assert s["net_pnl_when_enter"] == 5.0
    assert s["net_pnl_when_skip"] == -3.0


def test_unjoined_decisions_are_counted_not_silently_dropped():
    decisions = [_decision(1, "A", "ENTER", "ENTER", entry_order_id="999")]
    s = summarize(decisions, [])   # no matching trade yet
    assert s["unjoined"] == 1
    assert s["net_pnl_when_enter"] is None


def test_a_skip_with_no_order_id_is_not_counted_as_unjoined():
    # A refused candidate never became an order — that is not a joining
    # failure, it is the expected shape of a SKIP row.
    decisions = [_decision(1, "A", "SKIP", "SKIP", entry_order_id=None)]
    s = summarize(decisions, [])
    assert s["unjoined"] == 0


def test_by_verdict_counts_every_row_once():
    decisions = [_decision(1, "A", "ENTER", "ENTER"),
                 _decision(2, "A", "SKIP", "SKIP"),
                 _decision(3, "A", "SKIP", "UNKNOWN")]
    s = summarize(decisions, [])
    assert s["by_verdict"] == {"ENTER": 1, "SKIP": 1, "UNKNOWN": 1}


# ── reasons() composes from sub-judgements, not invented prose ──────────────

def test_reasons_are_derived_from_the_actual_sub_answers():
    r = _reasons("SKIP", conviction=2.8, looks_exhausted=0.9, regime_aligned=0.1)
    joined = " ".join(r)
    assert "exhausted" in joined
    assert "regime" in joined


def test_reasons_never_empty():
    r = _reasons("UNKNOWN", conviction=1.5, looks_exhausted=0.5, regime_aligned=0.5)
    assert r


# ── Wiring: auto_trader.py ───────────────────────────────────────────────────

def test_auto_trader_defaults_shadow_to_off():
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.__init__)
    assert "self.shadow = None" in src


def test_both_arms_are_wired_fire_and_forget():
    import bot.auto_trader as at
    src = inspect.getsource(at)
    assert src.count("decide_async(") == 2
    # Both call sites must be guarded the same way peer_eval already is —
    # never allowed to raise into the trading loop.
    i = src.index('bot_decision="SKIP", snap=snap)')
    j = src.index('bot_decision="ENTER"')
    assert "except Exception" in src[i:i + 100]
    assert "except Exception" in src[j:j + 200]


def test_the_enter_arm_carries_the_real_order_id():
    import bot.auto_trader as at
    src = inspect.getsource(at)
    assert 'entry_order_id=res.get("order_id")' in src


def test_entry_context_records_the_order_id_for_the_join():
    import bot.auto_trader as at
    src = inspect.getsource(at)
    assert '"entry_order_id": res.get("order_id")' in src


# ── Wiring: bot/api.py ───────────────────────────────────────────────────────

def test_the_shadow_route_exists_and_requires_auth():
    import bot.api as api
    src = inspect.getsource(api)
    i = src.index('@app.get("/api/shadow")')
    j = src.index("\n    @app.", i + 1)
    block = src[i:j]
    assert "Depends(_require_auth)" in block


def test_the_shadow_route_never_calls_the_model():
    import bot.api as api
    src = inspect.getsource(api)
    i = src.index('@app.get("/api/shadow")')
    j = src.index("\n    @app.", i + 1)
    block = src[i:j]
    assert "system_one" not in block
    assert "read_decisions" in block


def test_the_shadow_route_degrades_to_enabled_false_when_off():
    import bot.api as api
    src = inspect.getsource(api)
    i = src.index('@app.get("/api/shadow")')
    j = src.index("\n    @app.", i + 1)
    assert '"enabled": False' in src[i:j]


def test_the_shadow_route_never_lets_a_fault_become_a_500():
    import bot.api as api
    src = inspect.getsource(api)
    i = src.index('@app.get("/api/shadow")')
    j = src.index("\n    @app.", i + 1)
    block = src[i:j]
    assert "except Exception" in block
    assert '"error"' in block


# ── Config ────────────────────────────────────────────────────────────────

def test_shadow_is_off_by_default():
    from bot.config import BotConfig
    assert BotConfig().shadow_enabled is False


def test_main_wires_the_logger_only_when_enabled():
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent.joinpath("main.py").read_text(
        encoding="utf-8")
    assert "cfg.shadow_enabled" in src
    assert "ShadowDecisionLogger" in src
    assert "auto.shadow" in src
