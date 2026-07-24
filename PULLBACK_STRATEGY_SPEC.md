# Pullback Mode — Strategy Specification (v1)

This is the agreed spec for the new `STRATEGY=pullback` mode, as refined in
discussion. It supersedes the original spec document where they differ. The
breakout strategy (current five-filter continuation system) is unchanged and
remains the default; pullback is a separate, selectable mode.

## Design principles
- **Spot only.** No leverage, no futures. (Consistent with the whole system.)
- **Separate mode**, selected by `STRATEGY` env var. Breakout path untouched.
- **Testnet-first**, run as an isolated second container alongside live breakout.
- **Exits strictly as specified** — no profit-lock / trailing / peak-monitor
  machinery layered on. Clean mean-reversion/scalp exits only, so the strategy
  can be evaluated on its own terms.
- **Two regimes**, because a gaining coin and a falling coin are NOT the same
  setup and must not be treated identically.

## Timeframe & universe
- Timeframe: **3-minute** candles (configurable).
- Universe: Binance top gainers, spot USDT pairs.
- Liquidity: 24h volume > **10,000,000 USDT** (configurable).

## The universal "good entry price" gate (BOTH regimes)
The essence of the manual edge: **never enter at a relatively high price within
the candle**, and never at the tip of a rejection spike.

Compute candle position of the live price:
`position = (price - low) / (high - low)`  → 0.0 at low, 1.0 at high.

- **Require `position <= CANDLE_POS_MAX`** (default 0.5 — lower half). Refuses
  entries in the top of the candle.
- **Upper-wick veto:** if the upper wick is a large fraction of the candle range
  (`upper_wick / range >= UPPER_WICK_MAX`, default 0.40) AND price is high in the
  range, skip — that's a rejection spike ("narrow spike at the top").
- Entry is taken at/near a **fresh candle open**, when price is low in the new
  forming range — not chasing the top of a candle that already ran. This resolves
  the tension between "don't buy high" and "buy strength" (a strong green candle
  closes near its high; we enter on the next open, not that close).

## Regime 1 — Gainer (quick returns)
A coin clearly trending up. 24h low is NOT required (it's gaining; scalp on
momentum). Strict gates (the ones trusted from experience):
- **EMA(9) > EMA(21)** on 3m — critical, hard gate (optional small buffer so a
  coin sitting exactly on the crossover doesn't count; TBD default).
- **RSI in [55, 72]** (configurable band) on the confirmation candle.
- **RSI rising** — not merely above a level. RSI now must be above its value a few
  candles back AND ticking up on the confirmation candle, so a coin whose RSI has
  been declining (e.g. falling for ~15 min, now at 50) is rejected. (Exact
  lookback TBD — see open question.)
- **Volume expansion:** current volume > **1.3 × SMA(volume, 5)**.
- Universal good-price gate (above).

## Regime 2 — Dipper (outsized returns when right)
A coin currently falling BUT with strong volume. Here the 24h low is the key
support rationale — you do not buy a falling coin without a level that has held.
- **High 24h volume** (same liquidity floor, likely a higher bar in practice).
- **Proximity to 24h low:** price within `LOW_PROXIMITY_PCT` of the 24h low
  (configurable knob).
- **Signs of stabilizing:** RSI turning back up (the fall is pausing), volume
  present. This is where RSI-rising matters most — buying a dipper whose RSI is
  still falling is catching a knife.
- Universal good-price gate (above).
- **Wick-depth skip:** if the down-move implies a stop deeper than
  `MAX_WICK_STOP_PCT` (default ~1.5–2%), SKIP the trade entirely — too violent a
  flush is a breakdown, not a clean setup.

## Strict vs tunable (per stated priorities)
- **STRICT, load-bearing (trusted from experience):** EMA9>EMA21, RSI band,
  RSI-rising, volume expansion. These do not flex.
- **TUNABLE knobs (loosen on testnet for more data):** 24h-low proximity distance,
  wick-depth max, candle-position threshold, session windows.

## Session windows (UTC+3) — tunable, can widen on testnet
- 23:00–23:15 (US wind-down / funding)
- 04:00–06:00 (Tokyo / Asian open)
- 11:00–13:00 (London open / funding)
- 15:00–17:00 (pre-US / NY open)
(BTC-trend gating intentionally OMITTED for this strategy — no convincing
evidence for it in this approach. The existing BTC filter remains available but
off by default for pullback.)

## Exits (REVISED — reuse existing engine, per later decision)
User decision reversed the original "strict exits" call: to avoid losing large
gains (especially on dipper "outsized" moves), pullback REUSES the breakout bot's
existing exit engine rather than a fixed take-profit. Rationale: the trailing stop
+ profit lock + fast monitor are already built, tuned, and tested, and they do
exactly what's wanted (let runners run, protect gains, capture peaks).
- **Initial stop:** bounded wick SL (below the entry/dip candle low, capped by
  MAX_WICK_STOP_PCT; deeper → trade skipped at entry). Seeds the position's stop.
- **Then:** the existing trailing stop + profit lock take over (shared engine),
  so pullback trades protect gains and let runners run like breakout trades.
- **Timeout:** 5-candle (15 min on 3m) backstop, but ONLY force-closes a pullback
  trade that is FLAT or NEGATIVE. If the trade is in profit and trailing, the trail
  manages it — the timeout does not cut a runner (preserves "don't lose big gains").
  Strategy-aware: pullback uses pb_timeout_candles (5); breakout keeps
  max_hold_candles.
- Consequence: both strategies now exit similarly; only ENTRIES differ. The A/B
  comparison is therefore purely about entry quality — arguably the more useful
  comparison.


## Frequency intent
Maximize trades (especially on testnet for data) BUT only the right ones. Strict
gates stay strict; tunable knobs open up on testnet to increase flow.

## Deployment
- One shared image; `STRATEGY` selects mode.
- Two isolated containers on the server: existing **live-breakout** (unchanged) +
  new **testnet-pullback** (own port, own state volume, testnet creds).
- **Safety assertion:** if `TESTNET=false` with testnet creds (or misconfig),
  refuse to start — structural isolation, not conventional.

## RESOLVED
1. RSI-rising: confirmation candle RSI must be **above its value 3 candles back
   AND ticking up vs the prior candle** — blocks a genuine multi-candle decline
   (the "declining 15 min, now at 50" case). Lookback default 3 (tunable).
2. EMA9>EMA21 with a **small buffer** (EMA9 >= EMA21 * (1 + EMA_BUFFER_PCT),
   default ~0.05%) so near-crossover coins don't qualify. Other gates confirm the
   uptrend anyway.
