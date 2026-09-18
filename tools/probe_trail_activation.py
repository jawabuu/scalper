#!/usr/bin/env python3
"""
Does Binance honour `activationPrice` on a TRAILING_STOP_MARKET?

WHY THIS EXISTS
---------------
The guardian asks for an activation level and the exchange keeps the current
mark instead. STRK/USDT 2026-09-18 15:09, both trails on one position:

    sent 0.0351 (+0.347% vs mark)  ->  kept 0.0349784  (mark)
    sent 0.0349 (-0.238% vs mark)  ->  kept 0.0349817  (mark)

One above, one below, both replaced. `callbackRate` in the same payload IS
honoured. No error is returned.

ccxt is NOT the culprit: capturing the payload offline shows activationPrice
present and correct. But ccxt routes conditional orders on linear swaps to
`POST /fapi/v1/algoOrder` with algoType=CONDITIONAL (binance.py:6888), not to
`/fapi/v1/order`. That is also why these orders only ever appear in the algo
book.

So the open question is narrow: does the ALGO endpoint ignore activationPrice
while the PLAIN order endpoint honours it?

WHAT THIS DOES
--------------
Places the same trailing stop twice on an OPEN position — once through ccxt
(the algo path the bot uses today) and once directly through
`/fapi/v1/order` — then prints what each kept and cancels both.

Both orders are reduceOnly, so neither can open or increase a position. Both
are cancelled immediately. It refuses to run against a live account unless
--live is passed, and it never opens a position of its own: run it while one
is already open.

USAGE
-----
    python3 tools/probe_trail_activation.py                 # demo, auto-pick
    python3 tools/probe_trail_activation.py --symbol STRK/USDT:USDT
    python3 tools/probe_trail_activation.py --offset-pct 1.0
    python3 tools/probe_trail_activation.py --live          # deliberate

READING THE RESULT
------------------
    plain endpoint KEPT what was sent  -> the algo endpoint is the problem;
                                          place trails via /fapi/v1/order
    both substituted to mark           -> activationPrice is unsupported for
                                          trailing stops; arming at a level is
                                          not achievable and the design must
                                          change rather than be tuned
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


def build_exchange(demo: bool):
    """Mirror FuturesGuardian.__init__ exactly, proxy included."""
    key = _env("BINANCE_FUTURES_API_KEY", "BINANCE_API_KEY")
    sec = _env("BINANCE_FUTURES_API_SECRET", "BINANCE_API_SECRET")
    if not key or not sec:
        sys.exit("No API keys in the environment.")
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
    ap.add_argument("--live", action="store_true",
                    help="run against the LIVE account (default: demo)")
    a = ap.parse_args()

    demo = not a.live
    print(f"account: {'DEMO' if demo else 'LIVE'}")
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
    off = abs(a.offset_pct) / 100.0
    want = mark * (1 - off) if is_short else mark * (1 + off)
    want = float(ex.price_to_precision(symbol, want))

    print(f"position: {symbol} {pos.get('side')} qty={qty} mark={mark}")
    print(f"asking for activationPrice={want} "
          f"({'-' if is_short else '+'}{a.offset_pct}% from mark)")

    placed: list[tuple[str, str]] = []
    results: dict[str, bool] = {}

    # A) the path the bot uses today: ccxt -> /fapi/v1/algoOrder
    try:
        o = ex.create_order(
            symbol, "TRAILING_STOP_MARKET", side, qty, None,
            {"callbackRate": 0.5, "reduceOnly": True,
             "workingType": "MARK_PRICE", "activationPrice": want})
        oid, kept, rate = readback(o)
        results["algo"] = report(
            "A) ccxt create_order  ->  POST /fapi/v1/algoOrder", want, kept, mark)
        print(f"    callbackRate kept    : {rate}")
        if oid:
            placed.append((oid, "algo"))
    except Exception as e:
        print(f"\n  A) algo endpoint FAILED: {type(e).__name__}: {e}")

    # B) the plain order endpoint, bypassing ccxt's conditional routing
    try:
        raw = ex.fapiPrivatePostOrder({
            "symbol": ex.market(symbol)["id"],
            "side": side.upper(),
            "type": "TRAILING_STOP_MARKET",
            "quantity": ex.amount_to_precision(symbol, qty),
            "callbackRate": 0.5,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
            "activationPrice": want,
        })
        oid, kept, rate = readback({"info": raw})
        results["plain"] = report(
            "B) direct             ->  POST /fapi/v1/order", want, kept, mark)
        print(f"    callbackRate kept    : {rate}")
        if oid:
            placed.append((oid, "plain"))
    except Exception as e:
        print(f"\n  B) plain endpoint FAILED: {type(e).__name__}: {e}")
        print("     (if this is -1116 the order type is not valid there, "
              "which is itself the answer)")

    for oid, which in placed:
        try:
            ex.cancel_order(oid, symbol)
            print(f"\ncancelled {which} order {oid}")
        except Exception as e:
            print(f"\nCOULD NOT CANCEL {which} order {oid}: {e}")
            print("CANCEL IT BY HAND before leaving this position open.")

    print("\n" + "=" * 62)
    if results.get("plain"):
        print("The PLAIN endpoint honours activationPrice; the algo one does "
              "not.\n-> place trails via /fapi/v1/order and arming at a level "
              "works.")
    elif results.get("algo"):
        print("The ALGO endpoint honoured it here. Re-run — the earlier "
              "substitutions\nmay depend on the value or the symbol.")
    elif "plain" in results or "algo" in results:
        print("Neither endpoint honoured it. activationPrice is not usable "
              "for trailing\nstops on this account -> arming at a level is "
              "not achievable, and the\ndesign needs changing rather than "
              "tuning.")
    else:
        print("Both attempts failed — nothing was learned. Check the errors "
              "above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
