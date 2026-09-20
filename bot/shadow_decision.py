"""
Shadow decision log — what jev would have decided, recorded at decision time,
never acted on. Built against JEV-BRIEF.md; read that file before changing
this one, it is the spec.

WHAT THIS IS FOR
----------------
For every candidate the AUTO-ENTRY loop resolves — whether it enters or
refuses — ask jev a narrow, atomic set of questions about the setup and write
ONE line to `logs/shadow_decisions.jsonl`, BEFORE the outcome of the trade
(if any) is known. Later, `/api/shadow` joins these rows to the closed-trade
record by `entry_order_id` and reports net P&L split by what jev said versus
what the bot did — on BOTH arms, entries and refusals, per JEV-BRIEF.md §1:
without the refusal arm there is no counterfactual, and a real filter cannot
be told apart from a lucky one.

WHAT JEV IS ASKED, AND WHY IT IS NOT RSI AGAIN
-----------------------------------------------
JEV-BRIEF.md §1 is explicit that jev must add what the bot's own gates cannot
see — not recompute RSI, ATR or an EMA gap, which already work. The honest
starting point, investigated before writing this module (see the September
2026 finding this docstring records below), is that NOTHING flowing through
this codebase gives jev order-book depth, spread, book imbalance or trade-size
distribution — `CandidateStream` is mark-price-only, `entry_context` is
entirely indicator-derived, and there is no `fetch_order_book` call anywhere
in `bot/`. Building liquidity-aware questions today would mean inventing data
jev does not have.

So the four questions below ask for something different, and something a
threshold rule genuinely cannot do: a GESTALT read across several already-
computed structural fields at once — whether the taper/turn/breakout/advance-
volume shape reads as a move still expanding or one already exhausted, and
whether direction agrees with the broader regime. That is an interpretation
of the INTERACTION between indicators, not a restatement of any one of them.
It is explicitly NOT the liquidity judgement JEV-BRIEF.md is ultimately for —
every state payload says so in plain words to jev, and `inputs_seen` records
`"order_book": "UNAVAILABLE"` on every row — so this measurement is honest
about testing a narrower hypothesis than the brief's, not a substitute for it.
A live depth feed is the natural next step once this narrower case is proven
out or ruled out; see JEV-BRIEF.md §7 for the investigation that led here.

HARD CONSTRAINTS (JEV-BRIEF.md §5)
-----------------------------------
1. Never place, block, size or delay an order — enforced by construction:
   this module is only ever called with information about a decision the bot
   ALREADY made, on a background thread, after that decision is final.
2. No look-ahead — every question is asked against the exact scanner row the
   bot itself decided on, nothing fetched after the fact.
3. Log every input — `inputs_seen` records the full state sent to jev.
4. UNKNOWN is a valid answer — `jev_verdict="UNKNOWN"` is a real outcome
   (config broken, request failed, or jev's own answer was inconclusive),
   never silently coerced into ENTER or SKIP.
5. Deterministic given the same inputs, or record the seed — jev's own model
   id and the exact question wording are recorded on every row (`model`,
   `fingerprint`), so a changed prompt is visible in the data, not silently
   blended into it.
6. Fail-open, demonstrated by test, not just asserted in a docstring — see
   tests/test_shadow_decision.py.

NEVER CACHED
------------
"Should THIS candidate be entered right now" is a live judgement over the
current scanner row — identical inputs an hour apart describe two different
trades, unlike a fact about an asset's identity that holds for weeks. Every
call here is fresh.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("shadow_decision")

DEFAULT_PATH = "logs/shadow_decisions.jsonl"

# Most candidates get refused most cycles — that is normal for a strict
# scalper, not a fault — so logging every refusal uncapped would turn a
# single busy cycle into dozens of TypeSafe requests. Same defensive shape as
# bot/peer_eval.py's MAX_PER_MINUTE: cap the volume, drop the excess quietly,
# rather than let one noisy scan set the request budget.
MAX_PER_MINUTE = 30

# How long a candidate stays "already judged" — see _materially_changed.
#
# The cap above bounds volume but not REDUNDANCY: the first real sample was
# 1021 rows over 24 distinct symbols in minutes, CHILLGUY alone 106 times.
# The cap never fired, so nothing flagged it. Those rows are also not
# independent, which matters more than their cost: any statistic over them —
# and any later join to outcomes — would weight a handful of coins by how
# often the scanner happened to loop.
#
# 900s is a judgement call: long enough to collapse a scan loop, short enough
# that a genuinely evolving setup is re-asked within a candle or two. A real
# move re-keys immediately regardless of the window, since the key is state,
# not time.
DEDUP_WINDOW_SEC = 900.0

# Bounds on the dedup map. Age pruning fires at the soft limit; the hard
# ceiling evicts oldest-first when a single window holds more distinct states
# than pruning can clear, which age alone cannot bound.
_SEEN_SOFT_LIMIT = 512
_SEEN_HARD_LIMIT = 4096

# ── The questions ────────────────────────────────────────────────────────────
#
# One request, four atomic judgements, asked whether the bot enters or
# refuses. See the module docstring for why these four and not liquidity
# questions the codebase cannot yet support with real data.

_VERDICT_CRITERIA = {
    "enter":
        "The setup, taken purely on the structural readings given — trend "
        "shape, taper/turn/breakout structure, and regime alignment — looks "
        "like a move that is still developing and worth taking.",
    "skip":
        "The setup looks exhausted, structurally incoherent, or fighting the "
        "broader regime badly enough that entering now looks like chasing.",
    "unknown":
        "The readings given do not support a confident call either way. "
        "Choose this rather than guessing — a forced ENTER or SKIP here "
        "would be a guess dressed as a judgement.",
}

_CONVICTION_LEVELS = [
    "Mixed or contradictory. The structural signals disagree with each "
    "other about whether the move is expanding or fading.",
    "Weak. The signals lean toward a coherent picture but only slightly.",
    "Solid. Trend shape, taper/turn structure and regime line up on one "
    "side of the ENTER/SKIP question without real contradiction.",
    "Strong. Every structural reading given points the same way, cleanly.",
]

# ── ATOMIC QUESTIONS, COMPOSED IN CODE ──────────────────────────────────────
#
# There is NO "verdict" question. Asking one composite "would you take this
# trade?" hands the weighting to the model, and then a change of priorities is
# a prompt rewrite whose effect on past rows is unknowable.
#
# Each question below judges ONE thing. The verdict is assembled from them by
# `_compose_verdict` using weights that live in this file. When the balance
# needs to change, you edit a coefficient — and because every component is
# recorded raw on every row, the weights can be RE-FITTED OFFLINE against
# outcomes without re-running a single call.
#
# That last property is the point. It turns the shadow log from "was jev
# right?" into "which component carried the signal, and at what weight?".
_QUESTION_SPECS = {
    "conviction": (
        "score",
        "How cleanly do the structural readings in `candidate` agree with "
        "each other about the direction of this move? Judge agreement only — "
        "not whether the trade is a good one.",
        _CONVICTION_LEVELS,
    ),
    "looks_exhausted": (
        "noul",
        "Does `candidate` read as a move that has already run most of its "
        "course — an extended, decelerating spike being chased late — "
        "rather than one still gathering momentum? Judge this from the "
        "taper, turn and advance-volume structure together, not from `rsi` "
        "or `change_24h_pct` alone.",
        None,
    ),
    "regime_aligned": (
        "noul",
        "Is `candidate.side` aligned with the broader market regime "
        "described in `regime` (breadth and BTC's own trend), rather than "
        "against it?",
        None,
    ),
    "structure_intact": (
        "noul",
        "Do the taper, turn and breakout readings in `candidate` still "
        "describe an orderly move, rather than one that has become erratic? "
        "Judge the shape only — no order-book, spread or trade-size data is "
        "available, and that absence is a known limitation, not something to "
        "infer or guess at.",
        None,
    ),
}

# Code-owned weights. Positive means "argues FOR the trade".
# Starting values are deliberately plain — equal weight on the two readings
# with a clear direction, half on the softer ones. They are a starting point
# to be re-fitted from the logged components, NOT a tuned result.
_WEIGHTS = {
    "not_exhausted": 1.0,     # 1 - looks_exhausted
    "regime_aligned": 1.0,
    "structure_intact": 0.5,
    "conviction": 0.5,        # normalised from the 0-3 scale
}
# Above this, the composed score reads ENTER. Held here, not in the model.
_ENTER_THRESHOLD = 0.55


def _compose_verdict(conviction: float, looks_exhausted: float,
                     regime_aligned: float, structure_intact: float) -> tuple:
    """
    Assemble ENTER/SKIP from the atomic judgements.

    Returns (verdict, score). The score is the weighted mean on 0-1, recorded
    alongside the raw components so the threshold and the weights can both be
    re-examined against outcomes later.
    """
    parts = {
        "not_exhausted": 1.0 - float(looks_exhausted),
        "regime_aligned": float(regime_aligned),
        "structure_intact": float(structure_intact),
        "conviction": max(0.0, min(1.0, float(conviction) / 3.0)),
    }
    total = sum(_WEIGHTS.values()) or 1.0
    score = sum(parts[k] * _WEIGHTS[k] for k in parts) / total
    return ("ENTER" if score >= _ENTER_THRESHOLD else "SKIP"), round(score, 4)


def _fingerprint() -> str:
    blob = json.dumps(_QUESTION_SPECS, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


FINGERPRINT = _fingerprint()

VALID_VERDICTS = ("ENTER", "SKIP", "UNKNOWN")
VALID_BOT_DECISIONS = ("ENTER", "SKIP")


def _noul_confidence(noul: float) -> float:
    """
    A noul IS its own confidence — there is no separate field on NoulAnswer
    (only ScoreAnswer and ChoiceAnswer carry one). Per the API's own schema:
    values near 1 favour yes, near 0 favour no, and near 0.5 indicate
    uncertainty. So distance off the midpoint, doubled onto 0–1:

        0.50 -> 0.0    0.75 -> 0.5    0.95 -> 0.9    0.05 -> 0.9

    Symmetric on purpose: a confident NO is as usable a judgement as a
    confident YES. Only the middle is uninformative.

    Before v3.70.0 this did not exist and _parse read a `confidence` attribute
    off nouls with a silent `or 0.0` fallback. Three of the four questions are
    nouls, so min() took that zero and `confidence` was 0.0 on EVERY row ever
    written — a dead field that read like a measurement.
    """
    return round(2.0 * abs(float(noul) - 0.5), 4)


def _component_confidence(answer) -> float:
    """
    Confidence for one answer, by its actual type. Raises if neither shape is
    present: a component whose confidence cannot be established is a parse
    failure and belongs in the UNKNOWN path, NOT silently floored to zero.
    """
    noul = getattr(answer, "noul", None)
    if noul is not None:
        return _noul_confidence(noul)
    conf = getattr(answer, "confidence", None)
    if conf is None:
        raise ValueError(
            f"answer {type(answer).__name__} carries neither `noul` nor "
            f"`confidence` — cannot establish component confidence")
    return float(conf)


# How far each field must move from the LAST ASKED value to count as a new
# question. Absolute buckets were tried first and flap: a value drifting
# around a bucket edge re-keys on every crossing while nothing has changed.
# Measuring from the last asked value instead means drift must actually
# accumulate, so a candidate cannot re-ask itself by wobbling.
_MATERIAL_DELTA = {
    "rsi": 3.0,
    "atr_pct": 0.3,
    "ema_gap_pct": 0.3,
    "change_24h_pct": 2.0,
    "range_pos_24h": 0.08,
    "efficiency": 0.15,
    "taper_ratio": 0.15,
    "adv_price_pct": 0.8,
    "breadth_pct": 10.0,
    "btc_change_pct": 1.0,
    "htf_trend_pct": 1.0,
}

# Categoricals and bools: no noise floor, so any change is a new question.
_MATERIAL_FLAGS = (
    "gap_narrowing", "gap_rising", "er_direction", "adv_vol_trend",
    "adv_bars", "peak_vol_early", "turned_up", "bars_since_low",
    "tapering", "breakout", "brk_at_extreme", "brk_gap_widening",
)


def _state_signature(state: dict) -> tuple:
    """
    (flags, numerics) for one candidate — the two halves compared differently.

    Flags are exact-match; numerics are compared against the last ASKED value
    using _MATERIAL_DELTA. Returned rather than hashed because hysteresis
    needs the values themselves, not a digest.
    """
    c = state.get("candidate") or {}
    r = state.get("regime") or {}
    flags = tuple(c.get(k) for k in _MATERIAL_FLAGS)
    nums = {}
    for k in _MATERIAL_DELTA:
        v = c.get(k, r.get(k))
        if v is None:
            continue
        try:
            nums[k] = float(v)
        except (TypeError, ValueError):
            continue
    return flags, nums


def _materially_changed(prev: tuple, cur: tuple) -> bool:
    """
    True if this candidate is a genuinely different question from the last one
    asked about it.

    A field present now but absent when last asked (or vice versa) counts as
    changed: the scanner has started or stopped computing it, which is a real
    difference in what jev is being shown.
    """
    prev_flags, prev_nums = prev
    cur_flags, cur_nums = cur
    if prev_flags != cur_flags:
        return True
    if set(prev_nums) != set(cur_nums):
        return True
    for k, v in cur_nums.items():
        if abs(v - prev_nums[k]) >= _MATERIAL_DELTA[k]:
            return True
    return False


@dataclass
class ShadowDecision:
    """One row of logs/shadow_decisions.jsonl. Field names match JEV-BRIEF.md
    §5's schema, plus `conviction`/`looks_exhausted`/`regime_aligned`, which
    the brief's own §5's "reasons" free-text field cannot come from a System
    One model — jev returns typed probabilities, not prose (see `_reasons`)."""
    ts: float
    symbol: str
    side: str
    bot_decision: str          # "ENTER" | "SKIP"
    jev_verdict: str           # "ENTER" | "SKIP" | "UNKNOWN"
    confidence: float
    conviction: float
    looks_exhausted: float
    regime_aligned: float
    reasons: list
    entry_order_id: str | None
    model: str
    fingerprint: str
    inputs_seen: dict
    # Recorded RAW so the weights and threshold can be re-fitted offline
    # against outcomes, without re-running a single call. Defaults keep older
    # rows loadable.
    structure_intact: float = 0.0
    composed_score: float = 0.0
    weights: dict = None
    enter_threshold: float = 0.0

    def agrees_with_bot(self) -> bool | None:
        if self.jev_verdict == "UNKNOWN":
            return None
        return self.jev_verdict == self.bot_decision


def _reasons(verdict: str, conviction: float, looks_exhausted: float,
             regime_aligned: float) -> list:
    """
    jev returns typed probabilities, not the free-text explanations
    JEV-BRIEF.md's example schema shows ("thin ask-side depth..."). Those
    specific reasons assumed order-book data this build does not have (see
    the module docstring). This composes plain, code-authored phrases from
    the SUB-judgements instead, so the record still says why without
    inventing prose jev never generated.
    """
    out = []
    if looks_exhausted >= 0.65:
        out.append(f"reads as an exhausted move (p={looks_exhausted:.2f})")
    elif looks_exhausted <= 0.35:
        out.append(f"reads as still developing (p={1 - looks_exhausted:.2f})")
    if regime_aligned <= 0.35:
        out.append(f"fighting the broader regime (p_aligned={regime_aligned:.2f})")
    elif regime_aligned >= 0.65:
        out.append(f"aligned with the broader regime (p={regime_aligned:.2f})")
    if conviction <= 0.5:
        out.append(f"structural readings disagree (conviction={conviction:.1f}/3)")
    elif conviction >= 2.5:
        out.append(f"structural readings agree cleanly (conviction={conviction:.1f}/3)")
    if not out:
        out.append(f"verdict={verdict}, no reading stood out")
    return out


def _candidate_state(symbol: str, side: str, row: dict, snap: dict) -> dict:
    """
    Everything jev sees. Identity + the structural fields already computed
    by the scanner (never re-fetched — JEV-BRIEF.md §5 constraint 2, no
    look-ahead), plus an explicit, honest gap where liquidity data would go.
    """
    adv = row.get("advance") or {}
    taper = row.get("taper") or {}
    turn = row.get("turn") or {}
    breakout = row.get("breakout") or {}
    eff = row.get("efficiency") or {}
    return {
        "candidate": {
            "symbol": symbol,
            "side": side,
            "rsi": row.get("rsi"),
            "atr_pct": row.get("atr_pct"),
            "ema_gap_pct": row.get("ema_gap_pct"),
            "change_24h_pct": row.get("change_24h_pct"),
            "range_pos_24h": row.get("range_pos_24h"),
            "gap_narrowing": row.get("gap_narrowing"),
            "gap_rising": row.get("gap_rising"),
            "efficiency": eff.get("efficiency"),
            "er_direction": eff.get("er_direction"),
            "adv_vol_trend": adv.get("adv_vol_trend"),
            "adv_bars": adv.get("adv_bars"),
            "adv_price_pct": adv.get("adv_price_pct"),
            "peak_vol_early": adv.get("peak_vol_early"),
            "turned_up": turn.get("turned_up"),
            "bars_since_low": turn.get("bars_since_low"),
            "taper_ratio": taper.get("taper_ratio"),
            "tapering": taper.get("tapering"),
            "breakout": breakout.get("breakout"),
            "brk_at_extreme": breakout.get("at_extreme"),
            "brk_gap_widening": breakout.get("gap_widening"),
        },
        "regime": {
            "breadth_pct": snap.get("breadth_pct"),
            "btc_change_pct": snap.get("btc_change_pct"),
            "htf_trend_pct": row.get("htf_trend_pct"),
        },
        # Named explicitly rather than omitted, so a reader of the logged
        # state — or jev itself — sees the gap instead of inferring silence
        # as "not asked". See the module docstring / JEV-BRIEF.md §7.
        "order_book": "UNAVAILABLE — no live depth/spread feed exists in "
                      "this codebase (investigated 2026-09-19, see "
                      "JEV-BRIEF.md). Judge only on the fields given.",
    }


class ShadowDecisionLogger:
    """
    Fire-and-forget, mirroring bot/peer_eval.py's PeerEval.notify_entry
    exactly: a daemon thread per call, nothing the caller waits on, every
    exception swallowed at the boundary. Instrumentation that can affect a
    trade is worse than no instrumentation.
    """

    def __init__(self, path: str = DEFAULT_PATH, model: str = "jev-latest",
                 max_per_minute: int = MAX_PER_MINUTE, client=None,
                 dedup_window: float = DEDUP_WINDOW_SEC):
        self.path = Path(path)
        self.model = model
        self.max_per_minute = max_per_minute
        self._client = client              # injected in tests
        self._client_broken = False
        # 0 disables dedup entirely — the pre-v3.71.0 behaviour, kept because
        # it changes WHICH candidates get judged, not just how many.
        self.dedup_window = float(dedup_window)
        self._seen: dict = {}              # dedup key -> last-asked ts
        self.skipped_as_unchanged = 0
        self._lock = threading.Lock()
        self._recent: list[float] = []
        self.dropped_for_rate = 0

    def _state_is_new(self, symbol: str, side: str, state: dict) -> bool:
        """
        True if this candidate is a genuinely different question from the last
        one asked about it — see _materially_changed.

        Keyed on (symbol, side) rather than on a state hash, because
        hysteresis compares against what was last ASKED. That is what stops a
        drifting indicator re-asking itself every time it crosses a boundary.

        Bounded two ways. Age pruning alone is NOT enough: a wide scan can
        hold more symbols than the window ever expires, so the hard ceiling
        evicts oldest-first when pruning cannot help.
        """
        if self.dedup_window <= 0:
            return True
        sig = _state_signature(state)
        now = time.time()
        with self._lock:
            cutoff = now - self.dedup_window
            if len(self._seen) > _SEEN_SOFT_LIMIT:
                self._seen = {k: v for k, v in self._seen.items()
                              if v[0] > cutoff}
            if len(self._seen) > _SEEN_HARD_LIMIT:
                keep = sorted(self._seen.items(), key=lambda kv: kv[1][0],
                              reverse=True)[:_SEEN_HARD_LIMIT]
                self._seen = dict(keep)
            prev = self._seen.get((symbol, side))
            if prev is not None and prev[0] > cutoff and \
                    not _materially_changed(prev[1], sig):
                self.skipped_as_unchanged += 1
                return False
            self._seen[(symbol, side)] = (now, sig)
            return True

    def _rate_ok(self) -> bool:
        now = time.time()
        with self._lock:
            self._recent = [t for t in self._recent if now - t < 60.0]
            if len(self._recent) >= self.max_per_minute:
                self.dropped_for_rate += 1
                return False
            self._recent.append(now)
            return True

    def _get_client(self):
        # Breaker FIRST. It was previously reached only when self._client was
        # still None, so a client that constructed fine and then had its model
        # name rejected went on being handed out: the flag latched and gated
        # nothing.
        if self._client_broken:
            return None
        if self._client is not None:
            return self._client
        if not os.environ.get("TYPESAFE_API_KEY"):
            log.warning("shadow decision: TYPESAFE_API_KEY is not set — "
                        "disabled, nothing will be logged.")
            self._client_broken = True
            return None
        try:
            from typesafe_sdk import TypeSafeClient
            self._client = TypeSafeClient(model=self.model)
        except Exception as e:
            log.warning(f"shadow decision: no usable TypeSafe client ({e}) "
                        f"— disabled, nothing will be logged.")
            self._client_broken = True
            return None
        return self._client

    @staticmethod
    def _questions() -> dict:
        out = {}
        for qid, (kind, instructions, criteria) in _QUESTION_SPECS.items():
            q: dict = {"type": kind, "instructions": instructions}
            if criteria is not None:
                q["criteria"] = criteria
            out[qid] = q
        return out

    # ── the public call ───────────────────────────────────────────────────

    def decide_async(self, symbol: str, side: str, row: dict, *,
                     bot_decision: str, snap: dict | None = None,
                     entry_order_id: str | None = None) -> None:
        """
        Queue one jev judgement for this candidate and return immediately.
        Never raises, never blocks the caller — see the class docstring.
        """
        if bot_decision not in VALID_BOT_DECISIONS:
            log.debug(f"shadow decision: unknown bot_decision {bot_decision!r}")
            return
        client = self._get_client()
        if client is None:
            return
        # State is built BEFORE the rate check so an unchanged candidate does
        # not consume rate budget that a genuinely new one could have used.
        # Building it is pure dict work on data the scanner already computed —
        # no fetch, no look-ahead (JEV-BRIEF.md §5 constraint 2).
        state = _candidate_state(symbol, side, row, snap or {})
        if not self._state_is_new(symbol, side, state):
            if self.skipped_as_unchanged in (1, 100) or \
                    self.skipped_as_unchanged % 1000 == 0:
                log.info(f"shadow decision: {self.skipped_as_unchanged} "
                         f"candidate(s) skipped as materially unchanged "
                         f"within {self.dedup_window:.0f}s. Never affects the "
                         f"real decision.")
            return
        if not self._rate_ok():
            if self.dropped_for_rate in (1, 10) or self.dropped_for_rate % 100 == 0:
                log.warning(f"shadow decision: rate cap "
                            f"({self.max_per_minute}/min) reached — "
                            f"{self.dropped_for_rate} candidate(s) skipped "
                            f"so far. Never affects the real decision.")
            return
        t = threading.Thread(
            target=self._run, name="shadow-decision",
            args=(symbol, side, bot_decision, state, entry_order_id),
            daemon=True)
        t.start()

    def _run(self, symbol, side, bot_decision, state, entry_order_id):
        try:
            result = self._client.system_one(state, self._questions())
            rec = self._parse(symbol, side, bot_decision, state,
                              entry_order_id, result)
        except Exception as e:
            # A rejected model name is a CONFIG error, not a transient one:
            # retrying cannot fix it, and the client constructs fine because
            # the name is only validated server-side, per call. Without this
            # the loop fires up to max_per_minute doomed requests a minute
            # indefinitely, each one landing here as an UNKNOWN row.
            if "Unknown model" in str(e):
                self._client_broken = True
                log.error(f"shadow decision: SHADOW_MODEL={self.model!r} was "
                          f"rejected by the API — shadow logging DISABLED for "
                          f"this process. Valid names come from "
                          f"client.models.list(); 'jev' alone is the family, "
                          f"not an id.")
            else:
                log.warning(f"shadow decision: {symbol} judgement failed "
                            f"({type(e).__name__}: {e}) — logged as UNKNOWN.")
            rec = ShadowDecision(
                ts=time.time(), symbol=symbol, side=side,
                bot_decision=bot_decision, jev_verdict="UNKNOWN",
                confidence=0.0, conviction=0.0, looks_exhausted=0.0,
                regime_aligned=0.0,
                reasons=[f"jev call failed: {type(e).__name__}: {e}"],
                entry_order_id=entry_order_id, model=self.model,
                fingerprint=FINGERPRINT, inputs_seen=state)
        self._write(rec)

    def _parse(self, symbol, side, bot_decision, state, entry_order_id,
              result) -> ShadowDecision:
        conv = result.scores["conviction"]
        conviction = float(conv.score)
        exhausted = float(result.nouls["looks_exhausted"].noul)
        aligned = float(result.nouls["regime_aligned"].noul)
        intact = float(result.nouls["structure_intact"].noul)
        # THE VERDICT IS COMPOSED HERE, not asked. See _compose_verdict.
        verdict, score = _compose_verdict(conviction, exhausted, aligned,
                                          intact)
        # Confidence is the model's LEAST confident component, not an average:
        # a verdict assembled from four judgements is only as trustworthy as
        # its weakest input, and averaging would hide exactly the case worth
        # abstaining on.
        #
        # Nouls have no confidence FIELD — it is derived from how far off 0.5
        # they sit. See _noul_confidence.
        confs = [_component_confidence(o)
                 for o in (conv,
                           result.nouls["looks_exhausted"],
                           result.nouls["regime_aligned"],
                           result.nouls["structure_intact"])]
        return ShadowDecision(
            ts=time.time(), symbol=symbol, side=side,
            bot_decision=bot_decision, jev_verdict=verdict,
            confidence=round(min(confs) if confs else 0.0, 4),
            conviction=conviction, looks_exhausted=exhausted,
            regime_aligned=aligned, structure_intact=intact,
            composed_score=score, weights=dict(_WEIGHTS),
            enter_threshold=_ENTER_THRESHOLD,
            reasons=_reasons(verdict, conviction, exhausted, aligned),
            entry_order_id=entry_order_id,
            model=str(getattr(result, "model", self.model)),
            fingerprint=FINGERPRINT, inputs_seen=state)

    def _write(self, rec: ShadowDecision):
        line = json.dumps(rec.__dict__, ensure_ascii=False)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            log.warning(f"shadow decision: could not append to "
                        f"{self.path}: {e}")


# ── Reading, for /api/shadow (bot/api.py) ────────────────────────────────────
#
# Deliberately separate from the writer above: a GET must never call the
# model (JEV-BRIEF.md §5b constraint 4), so reading is pure file I/O plus the
# join against closed trades, nothing that touches TypeSafe.

def read_decisions(path=DEFAULT_PATH, limit: int = 500,
                   symbol: str | None = None, verdict: str | None = None,
                   since: float | None = None) -> list:
    """
    Newest-first rows from the JSONL log, matching JEV-BRIEF.md §5b's query
    params. Malformed lines are skipped, not fatal — an operator's tail -f
    truncating mid-write must not take the whole endpoint down.
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if symbol and d.get("symbol") != symbol:
                    continue
                if verdict and str(d.get("jev_verdict", "")).upper() != verdict.upper():
                    continue
                if since and float(d.get("ts", 0)) < since:
                    continue
                rows.append(d)
    except OSError:
        return []
    rows.sort(key=lambda d: d.get("ts", 0), reverse=True)
    return rows[:max(0, int(limit or 0))] if limit else rows


def summarize(decisions: list, closed_trades: list) -> dict:
    """
    JEV-BRIEF.md §5's evaluation, computed fresh from the rows every call so
    it can never drift from a separately maintained counter (§5b).

    Joined on `entry_order_id` — added to entry_context by auto_trader.py
    alongside this module's own hook, at the same point res["order_id"] is
    already in scope, so the id here is exactly the one Binance assigned.
    """
    by_order = {}
    for t in closed_trades:
        oid = (t.get("entry_context") or {}).get("entry_order_id")
        if oid:
            by_order[str(oid)] = t

    by_verdict: dict = {}
    net_by_verdict: dict = {}
    n_trials = len(decisions)
    unjoined = 0
    agree = 0
    disagree = 0
    for d in decisions:
        v = d.get("jev_verdict", "UNKNOWN")
        by_verdict[v] = by_verdict.get(v, 0) + 1
        a = ShadowDecision(**{k: d.get(k) for k in
                              ShadowDecision.__dataclass_fields__}
                          ).agrees_with_bot()
        if a is True:
            agree += 1
        elif a is False:
            disagree += 1
        oid = d.get("entry_order_id")
        if not oid:
            continue
        trade = by_order.get(str(oid))
        if trade is None:
            unjoined += 1
            continue
        pnl = trade.get("net_pnl_usdt")
        if pnl is None:
            continue
        net_by_verdict.setdefault(v, []).append(float(pnl))

    def _sum(key):
        vals = net_by_verdict.get(key)
        return round(sum(vals), 4) if vals else None

    denom = agree + disagree
    return {
        "n": n_trials,
        "n_trials": n_trials,
        "by_verdict": by_verdict,
        "agreement_with_bot": round(agree / denom, 4) if denom else None,
        "net_pnl_when_enter": _sum("ENTER"),
        "net_pnl_when_skip": _sum("SKIP"),
        "unjoined": unjoined,
    }
