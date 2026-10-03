"""
A TRADE PLAN: side, entry, stop, targets — derived from THIS coin's structure.

WHY THIS EXISTS

Today every trade gets the same global configuration: the same callback
formula, the same ATR-multiple stop, the same fail-fast. A coin at its daily
high with a 15% move and one 4% up in a tight range are treated identically.
The gates decide WHETHER to enter; nothing decides HOW to trade what was
entered.

The replay said that is the binding constraint, not the exits:

    rule                median      mean     TOTAL    win
    actual              +0.057    -0.153    -30.39    53%
    tiered 1%/2% SL2    +0.614    -0.196    -38.99    62%

Tiering produced the best median, the best win rate and the best paired
difference of any rule tested — and the worst total, because a FLAT 2% stop
fired on 34% of trades. A stop that fires on a third of trades is not marking
"this trade is wrong"; it is marking "price moved a bit".

THE LEVELS ARE ALREADY THERE AND INFORM NOTHING

`pct_below_24h_high`, `pct_above_24h_low`, `room_ahead_atr` and
`range_pos_24h` are computed on every candidate and used only to ACCEPT or
REFUSE it. None of them sets a stop or a target.

For a short at the 24h high, "wrong" means price RECLAIMING that high — which
is 0.4% away on one coin and 3% on another. That is what structural means: a
level that is different per coin because the coin is different.

HOW A PLAN IS SIZED

    stop     just beyond the extreme the setup faded, plus an ATR buffer so
             noise alone does not reach it
    targets  fractions of the ROOM AHEAD — the distance to the opposite 24h
             extreme, which is the move the thesis is actually predicting
    risk     derived, never assumed: a plan whose reward does not justify its
             risk is REFUSED rather than resized

A plan is a PROPOSAL. Scoring it against what happened afterwards is how we
learn whether structural levels beat ATR ones — the same method that produced
the RSI band and continuation findings, at no risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradePlan:
    """What the bot proposes for ONE candidate, in PRICE terms throughout."""
    symbol: str
    side: str                       # "long" | "short"
    entry: float
    stop: float
    targets: list                   # [(price, fraction), ...] in order
    # Everything below is derived, and recorded so a plan can be audited
    # after the fact rather than re-derived from stale inputs.
    stop_pct: float = 0.0           # distance to the stop, % of entry
    reward_pct: float = 0.0         # distance to the LAST target, % of entry
    rr: float = 0.0                 # reward / risk
    basis: str = ""                 # which extreme the stop was set against
    room_pct: float = 0.0           # distance to the opposite 24h extreme
    refused: str | None = None      # why this is not tradeable, if it is not
    # Targets as % of entry, which is the unit the plan is actually built in.
    # Absolute prices are derived and only present when an entry was supplied.
    target_pcts: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side,
            "entry": round(self.entry, 10),
            "stop": round(self.stop, 10),
            "targets": [[round(p, 10), round(f, 4)] for p, f in self.targets],
            "stop_pct": round(self.stop_pct, 4),
            "reward_pct": round(self.reward_pct, 4),
            "rr": round(self.rr, 3),
            "basis": self.basis,
            "room_pct": round(self.room_pct, 4),
            "target_pcts": [[g, f] for g, f in self.target_pcts],
            "refused": self.refused,
        }


@dataclass
class PlanConfig:
    """
    All distances in % of entry price. Nothing here is an ATR multiple: the
    point of a plan is that its levels come from the coin's own structure.
    """
    # How far BEYOND the faded extreme the stop sits, as a multiple of ATR.
    # The extreme itself is the thesis; the buffer is so ordinary noise does
    # not reach it. 0.5 ATR is half a typical candle.
    stop_buffer_atr: float = 0.5
    # A stop closer than this is inside the noise whatever the structure says.
    min_stop_pct: float = 0.30
    # A stop further than this is refused outright.
    #
    # THE TENSION THIS NUMBER EXPOSES. `AUTO_MAX_DIST_PCT=3` admits entries up
    # to 3% from the extreme being faded, so a stop placed BEYOND that extreme
    # is necessarily 3%+ plus the buffer. A structural stop on this
    # strategy's entries cannot be scalp-sized.
    #
    # At 3.5% of price the risk per trade is SEVEN TIMES the ~0.5% ATR stop
    # used today. The same margin therefore risks seven times as much, so a
    # plan-based trade must be SIZED FROM ITS OWN STOP, not from the global
    # risk budget — or it is simply a much bigger bet wearing a new name.
    #
    # Tightening AUTO_MAX_DIST_PCT is the other way to resolve it: entries
    # nearer the extreme give nearer stops.
    max_stop_pct: float = 3.50
    # Targets as fractions of the ROOM AHEAD, with the size closed at each.
    target_fracs: tuple = (0.25, 0.50)
    target_sizes: tuple = (0.34, 0.33)
    # Refuse a plan whose reward does not justify its risk. Sizing around a
    # bad plan is how a scalp becomes a hope.
    min_rr: float = 1.5
    # Refuse when there is no room: the thesis needs somewhere to go.
    min_room_pct: float = 0.50


def build_plan(row: dict, side: str, cfg: PlanConfig | None = None,
               entry: float | None = None) -> TradePlan:
    """
    Propose a plan for this candidate, or refuse it with a reason.

    WORKS IN PERCENTAGES, NOT PRICES. The scanner row carries
    `pct_below_24h_high` and `pct_above_24h_low` — the structure already
    expressed as distances — and `distance_to_extreme` uses them today. The
    first version of this read `row["price"]`, which does not exist: 338 of
    338 plans refused with "no entry price" and the generator never ran.

    Percentages are also the honest unit here. Everything else that matters
    on this bot — fees, ATR, callbacks, the payoff ratio — is a % of price,
    and absolute levels would have to be recomputed from a stale quote
    anyway. Absolute prices are filled in ONLY when an entry is supplied.

    Pure: no exchange, no state, no clock. A plan can be rebuilt from a
    shadow log months later and scored against what actually happened.
    """
    cfg = cfg or PlanConfig()
    short = str(side).lower().startswith("short")
    sym = str(row.get("symbol") or "")

    def _refuse(why: str) -> TradePlan:
        return TradePlan(symbol=sym, side=side, entry=float(entry or 0),
                         stop=0.0, targets=[], refused=why)

    to_high = row.get("pct_below_24h_high")
    to_low = row.get("pct_above_24h_low")
    if to_high is None or to_low is None:
        return _refuse("no 24h range — the structure is unknown")
    try:
        to_high, to_low = abs(float(to_high)), abs(float(to_low))
    except (TypeError, ValueError):
        return _refuse("24h range unreadable")

    atr_pct = row.get("atr_pct")
    buf = abs(float(atr_pct or 0)) * cfg.stop_buffer_atr

    # A short fades the 24h HIGH, so it is wrong when price reclaims it; the
    # room it has is the distance down to the 24h low. A long is the mirror.
    if short:
        stop_pct = to_high + buf
        room_pct = to_low
        basis = "24h high + %.2f ATR" % cfg.stop_buffer_atr
    else:
        stop_pct = to_low + buf
        room_pct = to_high
        basis = "24h low - %.2f ATR" % cfg.stop_buffer_atr

    if room_pct < cfg.min_room_pct:
        return _refuse("only %.2f%% room to the far extreme, need %.2f%%"
                       % (room_pct, cfg.min_room_pct))
    if stop_pct < cfg.min_stop_pct:
        # The structure saying the stop is very close is not a licence to use
        # it: inside the noise floor it fires on nothing in particular. That
        # is the 0.4% callback that cost QNT 19 ROI points.
        return _refuse("structural stop %.2f%% is inside the noise floor "
                       "(%.2f%%)" % (stop_pct, cfg.min_stop_pct))
    if stop_pct > cfg.max_stop_pct:
        return _refuse("structural stop %.2f%% exceeds the %.2f%% cap"
                       % (stop_pct, cfg.max_stop_pct))

    # Targets as fractions of the room the thesis predicts, in % of entry.
    tgt_pcts = [(room_pct * frac, size)
                for frac, size in zip(cfg.target_fracs, cfg.target_sizes)]
    if not tgt_pcts:
        return _refuse("no targets configured")

    reward_pct = tgt_pcts[-1][0]
    rr = reward_pct / stop_pct if stop_pct > 0 else 0.0
    if rr < cfg.min_rr:
        return _refuse("reward/risk %.2f below %.2f (%.2f%% reward on a "
                       "%.2f%% stop)" % (rr, cfg.min_rr, reward_pct, stop_pct))

    # Absolute levels only when an entry price is actually supplied.
    px = float(entry or 0)
    if px > 0:
        stop = px * (1 + stop_pct / 100) if short else px * (1 - stop_pct / 100)
        targets = [((px * (1 - g / 100) if short else px * (1 + g / 100)), f)
                   for g, f in tgt_pcts]
    else:
        stop = 0.0
        targets = [(0.0, f) for _g, f in tgt_pcts]

    return TradePlan(symbol=sym, side=side, entry=px, stop=stop,
                     targets=targets, stop_pct=stop_pct,
                     reward_pct=reward_pct, rr=rr, basis=basis,
                     room_pct=room_pct,
                     target_pcts=[(round(g, 4), f) for g, f in tgt_pcts])
