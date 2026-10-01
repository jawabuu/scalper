"""
effective_leverage must NEVER silently resolve to 1x on a leveraged venue.

LDO 2026-10-01. The position was partially reduced (qty 4699 -> 103) and the
margin field did not follow, so derived leverage collapsed from ~20 to 1.46 —
just under the 1.5 guard — and the reported field came back as 1. The
guardian then sized the armed trail as:

    "ARMED native trailing stop (callback 5.0% price = 5% ROI at 1x)"

instead of 0.3% at 20x. SIXTEEN TIMES TOO WIDE, with an activation 9.8% away
that could never be reached. The position ran on the fixed stop alone and
drifted 13 ROI points past where the dashboard said its stop was.

Returning 1.0 is not a safe default, it is the WORST one: every ROI figure
reads small and every stop distance reads wide, and nothing downstream can
tell the difference.
"""

import pytest

from bot.futures_guard import FuturesPosition


def _pos(qty, margin, leverage, declared=0.0, entry=0.4359):
    return FuturesPosition(symbol="LDO/USDT:USDT", side="long",
                           entry_price=entry, qty=qty, leverage=leverage,
                           margin=margin, declared_leverage=declared)


def test_a_healthy_position_uses_the_DERIVED_leverage():
    # 20100 x 0.01391 = 279.6 notional on 14.84 margin -> 18.8x
    p = _pos(20100, 14.84, 20, entry=0.01391)
    assert 18 < p.effective_leverage < 20


def test_the_LDO_COLLAPSE_no_longer_resolves_to_1x():
    """
    qty 103 at 0.4359 = 44.90 notional on 30.65 stale margin = 1.46 derived,
    just under the guard, with a reported field of 1.
    """
    bad = _pos(103, 30.65, 1)
    assert bad.effective_leverage == 1.0, "the old behaviour, for contrast"
    fixed = _pos(103, 30.65, 1, declared=20.0)
    assert fixed.effective_leverage == 20.0


def test_a_USABLE_reported_field_beats_the_declared_one():
    # The venue's own answer is better than the operator's default when it is
    # present and sane.
    p = _pos(103, 30.65, 10, declared=20.0)
    assert p.effective_leverage == 10.0


def test_derived_still_WINS_when_it_is_usable():
    # declared must never override a healthy derived figure, or a position
    # opened at a different leverage would be mis-sized.
    p = _pos(20100, 14.84, 1, declared=20.0, entry=0.01391)
    assert 18 < p.effective_leverage < 20


@pytest.mark.parametrize("declared", [0.0, None, 1.0, 1.4])
def test_an_UNUSABLE_declared_value_falls_through(declared):
    # A declared 1.0 or 1.4 is not a leveraged venue; it must not be trusted
    # over the existing behaviour.
    p = _pos(103, 30.65, 1, declared=declared or 0.0)
    assert p.effective_leverage == 1.0


def test_the_collapse_is_LOGGED_as_an_error():
    """
    It is silent by nature — every ROI reads small and every stop reads wide.
    An operator cannot see it without being told.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.fetch_positions)
    assert "LEVERAGE-COLLAPSE" in src
    assert "if eff < 1.5:" in src
    assert "partial reduction" in src


def test_the_declared_value_reaches_the_guard_config():
    import pathlib
    main = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text()
    i = main.index("gcfg = GuardConfig(")
    assert "declared_leverage=cfg.entry_target_leverage" in main[i:i + 600]


# ── The ROOT CAUSE: a partial fill corrupts the derived leverage ───────────

def test_margin_is_reconstructed_from_a_BELIEVABLE_leverage():
    """
    `notional / max(lev, 1)` forces derived leverage to EXACTLY lev — so when
    lev is itself 1, it manufactures a 1x position out of a 20x one.

    LDO 2026-10-01 17:09:59: order qty 4699 / notional 2069 / margin 103.47,
    but only 103 contracts had FILLED, so notional read 44.80 against a margin
    field still showing 101.90. Implied 0.44x -> reconstructed from lev=1 ->
    1.00x -> armed trail at 5.0% of price instead of 0.3%.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.fetch_positions)
    assert "use_lev = lev if lev >= 1.5 else (" in src
    assert "reconstructed = notional / use_lev" in src
    assert "notional / max(lev, 1)" not in src, "the collapsing form is gone"


def test_the_reconstruction_names_the_PARTIAL_FILL_cause():
    # The old message blamed cross-margin reporting, which sent the first
    # investigation down the wrong path entirely.
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.fetch_positions)
    assert "PARTIAL FILL" in src


def test_arming_is_DEFERRED_while_the_position_is_still_filling():
    """
    The armed trail is placed ONCE and never revised, so it must be sized
    against the position that will exist, not a fraction of it. LDO armed for
    103 of 4699 contracts; the other 4596 arrived into a position whose
    protection was sized for 2% of it.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._arm_at_entry)
    assert "NOT arming at entry" in src
    assert "pos.qty < float(sized_qty) * 0.9" in src


def test_the_sized_qty_is_recorded_on_the_handoff():
    import inspect
    from bot import futures_entry as fe
    src = inspect.getsource(fe.EntryService.execute)
    assert '"sized_qty": plan.qty' in src


def test_a_missing_sized_qty_does_not_block_arming():
    # Positions adopted without a handoff (operator-opened, restarts) have no
    # sized_qty. They must still get their trail.
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._arm_at_entry)
    assert "if sized_qty and pos.qty and" in src


def test_this_could_not_happen_before_maker_entry():
    # A TRAILING_STOP_MARKET entry fills in one go. Recorded so nobody
    # re-derives it: five symbols hit this within 28 minutes of the switch.
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._arm_at_entry)
    assert "maker_limit" in src or "maker limit" in src
