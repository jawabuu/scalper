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
    VALID_VERDICTS, _noul_confidence, _component_confidence, _triggers,
    BACKOFF_AFTER, BACKOFF_START_S, BACKOFF_MAX_S,
    _state_signature, _materially_changed,
)


# ── Fakes ────────────────────────────────────────────────────────────────────

class _Ans:
    """
    DEPRECATED, kept only for rows that do not care about answer shape.

    This fake accepted ANY keyword, so tests happily gave nouls a
    `confidence` attribute that the real NoulAnswer has never had. That is
    exactly how `confidence` stayed 0.0 on every production row while this
    file was green — the assertion read a field the test itself invented.
    Prefer _Noul and _Score below, which match the SDK schema.
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Noul:
    """Mirrors typesafe_sdk NoulAnswer: `noul` only. NO confidence field."""
    __slots__ = ("type", "noul")

    def __init__(self, noul):
        self.type = "noul"
        self.noul = float(noul)


class _Score:
    """Mirrors typesafe_sdk ScoreAnswer: `score` AND `confidence`."""
    __slots__ = ("type", "score", "confidence")

    def __init__(self, score, confidence):
        self.type = "score"
        self.score = float(score)
        self.confidence = float(confidence)


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
        self.scores = {"conviction": _Score(conviction, confidence)}
        # Nouls carry NO confidence — it is derived from distance off 0.5.
        self.nouls = {
            "looks_exhausted": _Noul(looks_exhausted),
            "regime_aligned": _Noul(regime_aligned),
            "structure_intact": _Noul(structure_intact)}


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


# ── The model NAME — never exercised before v3.69.0 ─────────────────────────
#
# Every test here injects a fake client, and the real one is constructed with
# a name the API only validates SERVER-side, per call. So SHADOW_MODEL=jev sat
# in config and .env.example through every green run and first failed in
# production, 400 "Unknown model: jev". These pin the two things that made it
# both possible and expensive.

def test_the_default_model_is_not_the_bare_family_name():
    # "jev" is the family. Every id the API accepts carries a suffix.
    from bot.config import BotConfig
    assert ShadowDecisionLogger().model != "jev"
    assert ShadowDecisionLogger().model.startswith("jev-")
    assert BotConfig().shadow_model.startswith("jev-")


def test_the_default_model_is_not_the_preview_channel():
    # Stability beats quality for a measurement rig: preview can move or
    # vanish mid-run, and rows either side would be silently incomparable.
    from bot.config import BotConfig
    assert "preview" not in ShadowDecisionLogger().model
    assert "preview" not in BotConfig().shadow_model


def test_a_rejected_model_name_disables_the_logger_instead_of_retrying(tmp_path):
    # A config error cannot be fixed by retrying. Before this, every candidate
    # fired another doomed request — up to SHADOW_MAX_PER_MINUTE a minute,
    # forever — each landing as an UNKNOWN row.
    path = tmp_path / "shadow.jsonl"
    c = _Client(raises=RuntimeError("400 Unknown model: jev"))
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    assert logger._client_broken, "a rejected model name must trip the breaker"
    logger.decide_async("Y/USDT:USDT", "long", _row(), bot_decision="SKIP")
    time.sleep(0.2)
    assert len(c.calls) == 1, "no further calls after the name was rejected"


def test_a_transient_failure_does_NOT_disable_the_logger(tmp_path):
    # The mirror of the above: a 502 is worth retrying and must not latch.
    path = tmp_path / "shadow.jsonl"
    c = _Client(raises=RuntimeError("502 Bad Gateway"))
    logger = ShadowDecisionLogger(path=str(path), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    assert not logger._client_broken
    logger.decide_async("Y/USDT:USDT", "long", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2)


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
    r = _Result(verdict="enter", confidence=0.9)
    # 0.55 is barely off the midpoint -> the weakest component by far.
    r.nouls["regime_aligned"] = _Noul(0.55)
    logger = ShadowDecisionLogger(path=str(path), client=_Client(r))
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="ENTER")
    assert _wait_for(lambda: path.exists() and path.read_text().strip())
    line = json.loads(path.read_text().strip().splitlines()[-1])
    assert line["confidence"] == 0.1, "must take the weakest, not the average"


# ── Confidence was a DEAD FIELD until v3.70.0 ───────────────────────────────
#
# NoulAnswer has no `confidence` attribute and never did. _parse read one with
# a silent `or 0.0` fallback, and three of the four questions are nouls, so
# min() took that zero: `confidence` was 0.0 on EVERY row ever written while
# reading like a real measurement. These pin the derivation and, more
# importantly, that the field VARIES.

def test_confidence_is_not_constant_across_differing_judgements(tmp_path):
    # The regression that matters. A field identical on every row measures
    # nothing, however plausible its value looks.
    path = tmp_path / "s.jsonl"
    for noul in (0.5, 0.75, 0.99):
        r = _Result(verdict="enter", confidence=0.9)
        r.nouls["regime_aligned"] = _Noul(noul)
        ShadowDecisionLogger(path=str(path), client=_Client(r)).decide_async(
            "X/USDT:USDT", "short", _row(), bot_decision="ENTER")
        assert _wait_for(lambda n=noul: path.exists() and
                         len(path.read_text().strip().splitlines()) ==
                         [0.5, 0.75, 0.99].index(n) + 1)
    got = [json.loads(l)["confidence"]
           for l in path.read_text().strip().splitlines()]
    assert len(set(got)) == 3, f"confidence must vary with the answers: {got}"
    assert got == sorted(got), "and rise as the nouls move off the midpoint"


def test_a_noul_at_the_midpoint_is_zero_confidence():
    # 0.5 is the model saying it cannot tell. That is the row worth flagging.
    assert _noul_confidence(0.5) == 0.0


def test_a_confident_no_counts_as_much_as_a_confident_yes():
    # Symmetric on purpose: only the middle is uninformative.
    assert _noul_confidence(0.05) == _noul_confidence(0.95)
    assert _noul_confidence(0.05) == 0.9


def test_an_answer_with_no_usable_confidence_raises_not_zeroes(tmp_path):
    # The `or 0.0` fallback is what let a missing field read as a real
    # measurement. A component whose confidence cannot be established is a
    # parse failure and belongs in the UNKNOWN path.
    with pytest.raises(ValueError):
        _component_confidence(_Ans(type="mystery"))


def test_nouls_carry_no_confidence_attribute():
    # Pins the fake to the SDK schema. If NoulAnswer ever gains a confidence
    # field, this fails and _component_confidence should be revisited.
    from typesafe_sdk._schemas.models import NoulAnswer, ScoreAnswer
    assert "confidence" not in NoulAnswer.model_fields
    assert "confidence" in ScoreAnswer.model_fields


# ── Dedup: the scanner re-offers the same candidate every cycle ─────────────
#
# First real sample: 1021 rows over 24 distinct symbols in minutes, CHILLGUY
# 106 times. The rate cap never fired — volume was never the problem,
# redundancy was. Non-independent rows corrupt any statistic over them and
# would manufacture false confidence in any later join to outcomes.

def test_an_unchanged_candidate_is_judged_once_not_every_cycle(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    for _ in range(10):
        logger.decide_async("CHILLGUY/USDT:USDT", "short", _row(),
                            bot_decision="SKIP", snap={"breadth_pct": 40})
    time.sleep(0.3)
    assert len(c.calls) == 1, "a re-offered, unchanged candidate is one question"
    assert logger.skipped_as_unchanged == 9


def test_a_materially_changed_candidate_is_judged_again(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    logger.decide_async("X/USDT:USDT", "short", _row(rsi=71.0),
                        bot_decision="SKIP")
    logger.decide_async("X/USDT:USDT", "short", _row(rsi=84.0),
                        bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2), \
        "a real move is a genuinely different question"


def test_float_noise_is_not_a_new_question(tmp_path):
    # A last-decimal wobble must not re-ask, or dedup collapses on live data.
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    logger.decide_async("X/USDT:USDT", "short", _row(rsi=71.0001),
                        bot_decision="SKIP")
    logger.decide_async("X/USDT:USDT", "short", _row(rsi=71.0002),
                        bot_decision="SKIP")
    time.sleep(0.3)
    assert len(c.calls) == 1


def test_different_symbols_never_collide(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    logger.decide_async("A/USDT:USDT", "short", _row(), bot_decision="SKIP")
    logger.decide_async("B/USDT:USDT", "short", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2)


def test_the_two_sides_of_one_symbol_are_separate_questions(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    logger.decide_async("X/USDT:USDT", "long", _row(), bot_decision="SKIP")
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2)


def test_dedup_runs_before_the_rate_cap(tmp_path):
    # Ordering matters: an unchanged candidate must not consume budget a
    # genuinely new one could have used.
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  max_per_minute=2)
    for _ in range(20):
        logger.decide_async("X/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    time.sleep(0.3)
    assert logger.dropped_for_rate == 0, "duplicates must not reach the cap"
    logger.decide_async("Y/USDT:USDT", "short", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2), \
        "budget must still be there for a new candidate"


def test_dedup_can_be_disabled_for_the_old_sampling(tmp_path):
    # It changes WHICH candidates are judged, so the old behaviour stays
    # reachable — a run compared against pre-v3.71.0 rows needs it.
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0)
    for _ in range(5):
        logger.decide_async("X/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 5)


def test_the_window_expires(tmp_path):
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0.2)
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="SKIP")
    time.sleep(0.35)
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2)


def test_the_seen_map_stays_bounded(tmp_path):
    # It is keyed by state, so an unbounded map would grow with every scan
    # for the life of the process.
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0.05)
    for i in range(700):
        logger.decide_async(f"S{i}/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    assert len(logger._seen) <= 700, len(logger._seen)
    # And a burst far past the hard ceiling still cannot grow without bound.
    for i in range(5000):
        logger.decide_async(f"T{i}/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    assert len(logger._seen) <= 4096, len(logger._seen)


def test_regime_is_compared_too_not_just_the_candidate():
    # The same coin in a different market is a different question.
    from bot.shadow_decision import _candidate_state
    a = _candidate_state("X/USDT:USDT", "short", _row(), {"breadth_pct": 10})
    b = _candidate_state("X/USDT:USDT", "short", _row(), {"breadth_pct": 90})
    assert _materially_changed(_state_signature(a), _state_signature(b))


def test_room_ahead_is_the_OPPOSITE_field_from_dist_to_extreme():
    """
    dist_to_extreme_pct measures the extreme the setup came FROM. Room ahead
    measures the one it is heading TOWARD. Confusing them would invert the
    whole test.
    """
    row = {"pct_above_24h_low": 4.0, "pct_below_24h_high": 0.5, "atr_pct": 1.0}
    short = _triggers(row, "short")
    long_ = _triggers(row, "long")
    assert short["room_ahead_pct"] == 4.0, "a short falls toward the 24h LOW"
    assert long_["room_ahead_pct"] == 0.5, "a long rises toward the 24h HIGH"


def test_room_is_also_expressed_in_ATR_units():
    # Config- and leverage-free, so it stays comparable across every setting
    # change in this investigation.
    t = _triggers({"pct_above_24h_low": 3.0, "atr_pct": 1.5}, "short")
    assert t["room_ahead_atr"] == 2.0


def test_room_is_None_rather_than_guessed_when_the_extreme_is_missing():
    t = _triggers({"atr_pct": 1.0}, "short")
    assert t["room_ahead_pct"] is None and t["room_ahead_atr"] is None


def test_recording_room_does_NOT_change_what_jev_is_shown():
    """
    It goes in TRIGGERS, not in the state. Adding it to the state would change
    the questions' inputs and the FINGERPRINT, invalidating every shadow row
    written so far.
    """
    from bot.shadow_decision import _candidate_state
    st = _candidate_state("X/USDT:USDT", "short",
                          _row(pct_above_24h_low=3.0), {})
    flat = json.dumps(st)
    assert "room_ahead" not in flat


def test_a_signature_survives_missing_fields():
    # Scanner rows are not guaranteed to carry every optional block.
    from bot.shadow_decision import _candidate_state
    sparse = _candidate_state("X/USDT:USDT", "short", {}, {})
    assert _state_signature(sparse)[1] == {}


def test_drift_must_ACCUMULATE_not_merely_cross_a_boundary(tmp_path):
    """
    The reason hysteresis replaced absolute buckets. On a fixed grid a value
    wobbling either side of an edge re-asks on every crossing; measured from
    the last ASKED value it cannot. Simulated over a drifting scan this was
    the difference between 15.6% and 3.1% of candidates judged.
    """
    c = _Client()
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c)
    for rsi in (71.0, 72.4, 71.1, 72.6, 71.3, 72.9):     # wobble, no trend
        logger.decide_async("X/USDT:USDT", "short", _row(rsi=rsi),
                            bot_decision="SKIP")
    time.sleep(0.3)
    assert len(c.calls) == 1, "wobble around a threshold is not a new question"
    logger.decide_async("X/USDT:USDT", "short", _row(rsi=75.0),
                        bot_decision="SKIP")
    assert _wait_for(lambda: len(c.calls) == 2), "real drift IS a new question"


def test_a_field_appearing_or_vanishing_counts_as_changed():
    # The scanner starting or stopping computing something changes what jev
    # is shown, even if every shared field is identical.
    from bot.shadow_decision import _candidate_state
    full = _candidate_state("X/USDT:USDT", "short", _row(), {})
    gone = _candidate_state("X/USDT:USDT", "short", _row(rsi=None), {})
    assert _materially_changed(_state_signature(full), _state_signature(gone))


# ── Backoff on provider outages ─────────────────────────────────────────────
#
# 2026-09-21 07:23-07:27: TypeSafe returned 503 "no healthy upstream", then
# 529 "high traffic ... try again later", then read timeouts, then recovered
# on its own. Throughout, every candidate kept firing into a service
# explicitly asking to be left alone, one UNKNOWN row and one WARNING each.
#
# Politeness and log noise, NOT data integrity — the resolver drops UNKNOWN
# rows and the shadow path is advisory.

def _fail(msg):
    return RuntimeError(msg)


def test_repeated_server_failures_pause_judgements(tmp_path):
    c = _Client(raises=_fail("503 no healthy upstream"))
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0)
    for i in range(12):
        logger.decide_async(f"S{i}/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    time.sleep(0.4)
    assert len(c.calls) <= BACKOFF_AFTER + 2, \
        f"should stop calling after ~{BACKOFF_AFTER} failures, made {len(c.calls)}"
    assert logger.dropped_for_backoff > 0


def test_a_CONFIG_error_does_not_trigger_backoff_it_LATCHES(tmp_path):
    # 400 Unknown model is unfixable by waiting; it must disable outright.
    c = _Client(raises=_fail("400 Unknown model: jev"))
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0)
    logger.decide_async("X/USDT:USDT", "short", _row(), bot_decision="SKIP")
    assert _wait_for(lambda: logger._client_broken)
    assert logger._consecutive_failures == 0, "a 4xx is not a transient failure"


def test_a_400_that_is_NOT_unknown_model_is_also_not_transient():
    logger = ShadowDecisionLogger(path="/dev/null")
    assert not logger._is_transient(_fail("400 Bad Request: malformed state"))
    assert not logger._is_transient(_fail("401 Unauthorized"))


def test_the_outage_statuses_ARE_transient():
    logger = ShadowDecisionLogger(path="/dev/null")
    for msg in ("503 no healthy upstream",
                "529 We are currently experiencing high traffic",
                "Request timed out (timeout=10.0).",
                "502 Bad Gateway", "500 Internal Server Error"):
        assert logger._is_transient(_fail(msg)), msg


def test_one_good_response_clears_the_pause(tmp_path):
    c = _Client(raises=_fail("529 high traffic"))
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0)
    for i in range(8):
        logger.decide_async(f"S{i}/USDT:USDT", "short", _row(),
                            bot_decision="SKIP")
    time.sleep(0.3)
    assert logger._backoff_until > 0
    logger._note_success()
    assert logger._backoff_until == 0.0
    assert logger._consecutive_failures == 0
    assert logger._backoff_s == BACKOFF_START_S, "escalation resets too"


def test_the_pause_escalates_then_caps(tmp_path):
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"),
                                  client=_Client(), dedup_window=0)
    seen = []
    for _ in range(12):
        logger._consecutive_failures = BACKOFF_AFTER
        logger._backoff_until = 0.0
        logger._note_failure(_fail("503 no healthy upstream"))
        seen.append(logger._backoff_s)
    assert seen[0] < seen[1], "must escalate"
    assert max(seen) <= BACKOFF_MAX_S, "and cap"


def test_in_flight_threads_do_not_extend_an_active_pause(tmp_path):
    # Threads dispatched before the pause began land afterwards; letting each
    # one re-arm the backoff would stretch a 30s pause indefinitely.
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"),
                                  client=_Client(), dedup_window=0)
    logger._consecutive_failures = BACKOFF_AFTER
    logger._note_failure(_fail("503 no healthy upstream"))
    first = logger._backoff_until
    for _ in range(5):
        logger._note_failure(_fail("503 no healthy upstream"))
    assert logger._backoff_until == first


def test_backoff_never_touches_the_real_decision(tmp_path):
    # The whole point: a provider outage must be invisible to trading.
    c = _Client(raises=_fail("503 no healthy upstream"))
    logger = ShadowDecisionLogger(path=str(tmp_path / "s.jsonl"), client=c,
                                  dedup_window=0)
    for i in range(20):
        assert logger.decide_async(f"S{i}/USDT:USDT", "short", _row(),
                                   bot_decision="SKIP") is None


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
