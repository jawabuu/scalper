#!/usr/bin/env python3
"""
Which parameter stops Binance honouring `activationPrice`?

WHAT IS ALREADY SETTLED
-----------------------
* The exchange DOES honour it. A trailing stop placed from the Binance UI on
  demo kept `Activation Price <= 0.0200000` with the market at 0.02312 —
  13.5% away, stored exactly, and it opened as a CONDITIONAL order.
* So `algoType=CONDITIONAL` is not the discriminator.
* `/fapi/v1/order` is not an option: -4120, "Order type not supported for this
  endpoint. Please use the Algo Order API endpoints instead."
* Every order the BOT places has its activation replaced with the current
  price. Three samples, no error returned.

WHAT DIFFERS between the UI order and ours, still untested:

  1. workingType  — we send MARK_PRICE. The UI form shows a `Last` selector
                    beside the activation field.
  2. reduceOnly   — all three substituted orders were reduceOnly=True. The UI
                    order was not, and neither is the bot's ENTRY trail, which
                    also keeps its activation (ONE: Activation Price >=
                    0.0015534).

TEST ORDER
----------
Step 1 costs nothing and changes one variable:

    python3 tools/probe_trail_activation.py --working-type CONTRACT_PRICE

  kept  -> workingType is the cause; set it on the trail and arming works
  lost  -> go to step 2

Step 2 varies reduceOnly instead. This one CAN open a position if it
activates, so it uses a minimal quantity and an activation far from market,
and cancels immediately:

    python3 tools/probe_trail_activation.py --no-reduce-only --far-pct 10

Each run places ONE order, reads the activation back, and cancels it through
the ALGO endpoint (`cancel_order` searches the regular book and returns -2011
on these — that is what stranded an order on the first run).

Demo unless --live.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import ccxt
except ImportError:
    sys.exit("ccxt is not importable — run this inside the bot container.")


def _env(*names, default=""):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def resolve_demo() -> bool:
    """
    The guardian's demo switch, exactly as bot/config.py resolves it:
    GUARDIAN_DEMO, else GUARDIAN_TESTNET, else TESTNET, defaulting True.
    """
    return _env_bool("GUARDIAN_DEMO",
                     _env_bool("GUARDIAN_TESTNET",
                               _env_bool("TESTNET", True)))


def build_exchange(demo: bool):
    """Mirror FuturesGuardian.__init__ exactly, proxy included."""
    # bot/config.py:_resolve_credentials — the suffix follows TESTNET, not the
    # guardian's demo flag, so read it the same way rather than guessing.
    suffix = "TEST" if _env_bool("TESTNET", True) else "LIVE"
    key = _env(f"BINANCE_API_KEY_{suffix}")
    sec = _env(f"BINANCE_API_SECRET_{suffix}")
    if not key or not sec:
        have = sorted(k for k in os.environ
                      if k.startswith("BINANCE_API_KEY"))
        sys.exit(
            f"No API keys: BINANCE_API_KEY_{suffix} / "
            f"BINANCE_API_SECRET_{suffix} are not set.\n"
            f"TESTNET={os.environ.get('TESTNET')!r} selects that suffix.\n"
            f"Key vars present: {have or 'none'}")
    params = {
        "apiKey": key, "secret": sec, "enableRateLimit": True,
        "options": {"defaultType": "future",
                    "fetchOpenOrders": {"warnWithoutSymbol": False}},
    }
    proxy = _env("SOCKS_PROXY", "HTTPS_PROXY", "HTTP_PROXY")
    if proxy:
        params["proxies"] = {"http": proxy, "https": proxy}
    ex = ccxt.binanceusdm(params)
    if demo:
        try:
            ex.enable_demo_trading(True)
        except AttributeError:
            ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex


def open_position(ex, want: str | None):
    for p in ex.fetch_positions() or []:
        try:
            amt = float(p.get("contracts") or 0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt <= 0:
            continue
        if want and p.get("symbol") != want:
            continue
        return p
    return None


def readback(order: dict) -> tuple[str | None, str | None, str | None]:
    info = (order or {}).get("info") or {}
    return (str(info.get("orderId") or (order or {}).get("id") or ""),
            info.get("activatePrice") or info.get("activationPrice"),
            info.get("priceRate") or info.get("callbackRate"))


def report(label: str, sent, kept, mark) -> bool:
    """True when the exchange kept what was asked for."""
    print(f"\n  {label}")
    print(f"    sent activationPrice : {sent}")
    print(f"    kept activationPrice : {kept}")
    print(f"    mark at placement    : {mark}")
    if kept in (None, ""):
        print("    VERDICT: no activation returned — inconclusive")
        return False
    try:
        d = (float(kept) - float(sent)) / float(sent) * 100
        dm = (float(kept) - float(mark)) / float(mark) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        print("    VERDICT: unparseable — inconclusive")
        return False
    print(f"    kept vs sent         : {d:+.4f}%")
    print(f"    kept vs mark         : {dm:+.4f}%")
    if abs(d) < 0.01:
        print("    VERDICT: HONOURED")
        return True
    if abs(dm) < 0.05:
        print("    VERDICT: SUBSTITUTED WITH MARK")
    else:
        print("    VERDICT: changed, and not to mark either")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--offset-pct", type=float, default=0.5,
                    help="how far from mark to ask for, away from profit")
    ap.add_argument("--key", default="activationPrice",
                    choices=["activationPrice", "activatePrice",
                             "triggerPrice", "stopPrice"],
                    help="the request field name to try")
    ap.add_argument("--working-type", default="MARK_PRICE",
                    choices=["MARK_PRICE", "CONTRACT_PRICE"])
    ap.add_argument("--no-reduce-only", action="store_true",
                    help="step 2: CAN open a position — minimal qty, far "
                         "activation, cancelled at once")
    ap.add_argument("--far-pct", type=float,
                    help="override --offset-pct, for a deliberately distant "
                         "activation")
    ap.add_argument("--live", action="store_true",
                    help="run against the LIVE account (default: demo)")
    a = ap.parse_args()

    # Default to whatever this container is actually configured as, so the
    # probe cannot silently talk to the wrong endpoint. --live is still
    # required to run against a live account.
    demo = resolve_demo()
    if not demo and not a.live:
        sys.exit("This container is configured LIVE. Re-run with --live if "
                 "that is deliberate.")
    if a.live:
        demo = False
    print(f"account: {'DEMO' if demo else 'LIVE'}  "
          f"(GUARDIAN_DEMO/TESTNET resolved)")
    ex = build_exchange(demo)

    pos = open_position(ex, a.symbol)
    if not pos:
        print("\nNo open position. This probe never opens one — reduceOnly "
              "orders need a position to attach to.\nRun it while the bot "
              "holds something, or pass --symbol for a specific one.")
        return 2

    symbol = pos["symbol"]
    qty = abs(float(pos.get("contracts") or 0))
    is_short = str(pos.get("side") or "").lower() == "short"
    side = "buy" if is_short else "sell"
    mark = float(pos.get("markPrice") or 0) or float(
        ex.fetch_ticker(symbol).get("last"))

    # Away from the mark on the profitable side, which is where an arm level
    # sits: below mark for a short, above for a long.
    off = abs(a.far_pct if a.far_pct else a.offset_pct) / 100.0
    want = mark * (1 - off) if is_short else mark * (1 + off)
    want = float(ex.price_to_precision(symbol, want))

    print(f"position: {symbol} {pos.get('side')} qty={qty} mark={mark}")
    shown = a.far_pct if a.far_pct else a.offset_pct
    print(f"asking for {a.key}={want} "
          f"({'-' if is_short else '+'}{shown}% from mark)")

    placed: list[tuple[str, str]] = []
    results: dict[str, bool] = {}

    # ccxt builds the payload for /fapi/v1/order and posts it to
    # /fapi/v1/algoOrder — different endpoints, different schemas. Binance
    # IGNORES unknown parameters rather than rejecting them, which is exactly
    # what we see: accepted, no error, default applied. And the RESPONSE names
    # the field `activatePrice`, not `activationPrice`.
    params = {"callbackRate": 0.5, "workingType": a.working_type,
              a.key: want}
    if not a.no_reduce_only:
        params["reduceOnly"] = True
    use_qty = qty if not a.no_reduce_only else min(qty, float(
        ex.amount_to_precision(symbol, qty * 0.01)) or qty)
    print(f"params: key={a.key} workingType={a.working_type} "
          f"reduceOnly={not a.no_reduce_only} qty={use_qty}")
    if a.no_reduce_only:
        print("  NOTE: not reduceOnly — this order CAN open a position if it "
              "activates.\n        Minimal quantity, activation far from "
              "market, cancelled immediately.")
    try:
        o = ex.create_order(symbol, "TRAILING_STOP_MARKET", side, use_qty,
                            None, params)
        oid, kept, rate = readback(o)
        results["kept"] = report(
            f"key={a.key} workingType={a.working_type}",
            want, kept, mark)
        print(f"    callbackRate kept    : {rate}")
        if oid:
            placed.append((oid, "algo"))
    except Exception as e:
        print(f"\n  placement FAILED: {type(e).__name__}: {e}")

    for oid, which in placed:
        done = False
        try:
            ex.fapiPrivateDeleteAlgoOrder({"algoId": oid})
            print(f"\ncancelled algo order {oid}")
            done = True
        except Exception as e:
            print(f"\nalgo cancel failed for {oid}: {e}")
        if not done:
            try:
                ex.cancel_order(oid, symbol)
                print(f"cancelled {oid} via the regular book")
            except Exception as e:
                print(f"COULD NOT CANCEL {oid}: {e}")
                print("CANCEL IT BY HAND — it is resting on your position.")

    print("\n" + "=" * 62)
    if results.get("kept"):
        print(f"HONOURED with workingType={a.working_type} "
              f"reduceOnly={not a.no_reduce_only}.\n"
              f"-> that is the setting the trail needs.")
    elif "kept" in results:
        print(f"STILL SUBSTITUTED with workingType={a.working_type} "
              f"reduceOnly={not a.no_reduce_only}.\n"
              f"-> rule this combination out and vary the other parameter.")
    else:
        print("Nothing was learned — see the error above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
