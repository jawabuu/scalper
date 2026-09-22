# Handover — Binance futures scalping bot

**v3.75.0**, 2026-09-21. Supersedes the old HANDOVER.md, which had drifted for
three months because it was never committed. Keep this one in git.

v3.80.0 RECORDS room-ahead (never gates on it) to test one external claim.
v3.79.3 fixes a regime guard that switched itself off once a third cohort
appeared — exactly when it was needed.
v3.79.2 shows non-floor cohorts and warns when one is ATR-SELECTED.
v3.79.1 adds never_green to that tool.
v3.79.0 adds tools/compare_callback_mult.py — the WITHIN-INSTANCE before/after
for a multiplier change. Run it BEFORE deploying to capture the baseline.
v3.78.2 documents the DEMO-vs-LIVE STRUCTURAL FACTOR — read it before
porting any ATR-derived setting between instances.
v3.78.1 WITHDRAWS the v3.78.0 diagnosis — REJECTED meant "invalidated at
close", not "refused at placement". The verification code stays as a safety
net. See the correction in that section.
v3.77.3 adds the floor trail to the GIVE-BACK resting dict.
v3.77.2 stops the floor trail falling back to a derived activation.
v3.77.1 fixes two gaps found in the first demo run of the floor trail.
v3.77.0 adds the DORMANT FLOOR TRAIL (off by default) — the break-even
promise no longer depends on a poll.
v3.76.1 surfaces a missing break-even floor on the dashboard.
v3.76.0 backs the shadow logger off during provider outages.

**START AT "THE OPEN PROBLEM: entry timing"** — that is the live work. The
rest is settled, shelved, or a record of reversals.

v3.75.0 RECALIBRATES the outcome horizons to actual hold time — every earlier
trigger conclusion was scored at the wrong timescale.
v3.74.0 adds PATH capture (adverse/favourable excursion) to the outcome
resolver — endpoints alone cannot express the entry-timing goal.
v3.73.6 documents peak_roi being UNRELIABLE (below) — read it before using
that column for anything.
v3.73.5 routes the outcome resolver through SOCKS_PROXY and records the FIRST
real trigger results (below).
v3.73.4 DEFAULTS THE VOLATILITY FLOOR OFF — 395 historical trail exits do
not support it (SETTLED #6). Both containers can drop the env var entirely.
v3.73.3 adds the missing sweep regression tests and a SETTLED section
recording which conclusions were reversed and why — read it before reopening
any of them.
v3.73.2 documents the AUTO_CALLBACK_ATR_MULT asymmetry — docs only.
v3.73.1 fixes arm-at-entry against the deferred arm level.
v3.73.0 records competing entry TRIGGERS on every shadow row.
v3.72.1 chains the multiplier default. v3.72.0 changes `bot/futures_guard.py`, `bot/futures_guardian.py`,
`bot/config.py`, `main.py`, `.env.example`, `tests/test_futures_guard.py`,
`tests/test_futures_guardian.py`. 1324 tests passing (was 1268 at v3.68.0).
Two new env settings, both default ON:
`GUARD_TRAIL_CALLBACK_ATR_MULT` and `GUARD_MIN_TRAIL_LOCK_ROI`.

**v3.72.0 CHANGES LIVE TRADING BEHAVIOUR** — armed trails are wider and arm
later on volatile coins. Read that entry before deploying.

v3.69.0-v3.71.0 are one separate investigation into the shadow log: v3.69.0
made it reach the API at all, v3.70.0 made what it records mean something,
v3.71.0 made the rows independent and gave them outcomes to measure against.

The v3.68.0 notes follow from "The finding".

---

## Arm-at-entry had never survived its own first cycle (pre-existing)

Found while auditing, not introduced by this work. `_cancel_superseded_stops`
protected `floor_stop_id` and `adaptive_trail_id` but NOT `native_trail_id`.
The armed trail is placed at adoption, lands in `_all_stop_ids`, and the next
fixed-stop placement swept it — LSK live 2026-09-20 08:03:23 four seconds
after placement, demo 07:50:12 the same. Both instances, every position.

The guardian's own comment: arm-at-entry "has therefore never survived its own
first cycle, which is why the poll-driven ratchet has gone on doing all the
work."

**This matters for reading historical demo data.** Trades from before that fix
were protected by the RATCHET, not by the armed trail, whatever `armed` says
in the journal. Do not compare give-back or capture across that boundary.

Same class as SOLV 19:26:49, where the sweep read `self._states` while
`manage_position` mutates a LOCAL state, so a trail placed earlier in the same
cycle was invisible and got cancelled one second after placement.

v3.73.3 adds the two regression tests that were missing:
`test_the_ARM_AT_ENTRY_trail_is_never_swept_as_superseded` and
`test_a_trail_placed_EARLIER_IN_THIS_CYCLE_is_protected`. The sweep's
internals and the `armed_replaced_stop` flag were already covered; the
protected-set membership for `native_trail_id` was not.

---

## The outcome resolver must use SOCKS_PROXY (v3.73.5)

Binance answers 451 "restricted location" from the VPS's region. ccxt hits
`exchangeInfo` during market loading BEFORE any candle fetch, so every row
failed identically and logged its own warning — one failure looking like
2000.

`tools/resolve_shadow_outcomes.py` now reads `SOCKS_PROXY` the way
`bot/engine.py` does (production: `http://gluetun:8888`), calls
`load_markets()` up front, and exits with one clear message if that fails.
Region and connectivity problems are fatal for the pass, not per-row.

`resolve()` is idempotent, so an aborted run loses nothing — re-run it.

---

## Room ahead — recorded, NOT gating (v3.80.0)

From an external article (casper_smc, "How to Scalp Like the Top 1% of
Traders", 2026-09-22). Volume-profile / auction-market reasoning, one
testable claim:

> Before entering, compare the room to the next level against the distance to
> invalidation, and skip the trade when the obstacle is closer than the stop.

The bot has no equivalent. `AUTO_MAX_DIST_PCT=3.0` requires being WITHIN 3%
of the extreme — a proximity filter, the opposite of a room check.

### What is recorded

    room_ahead_pct   distance to the 24h extreme the trade heads TOWARD
    room_ahead_atr   the same in units of the coin's own noise

In `entry_context` (auto_trader) and in shadow `triggers`. **Deliberately in
TRIGGERS, not in the state jev sees** — putting it in the state would change
the questions' inputs and the FINGERPRINT, invalidating every shadow row
written so far.

**Note it is the OPPOSITE field from `dist_to_extreme_pct`**, which measures
the extreme the setup came FROM. A short falls toward the 24h LOW
(`pct_above_24h_low`); a long rises toward the HIGH. Confusing them inverts
the whole test, so there is a test pinning it.

`summarize()` bands it as `<1 ATR / 1-2 / 2-4 / >4` — ATR units, so it stays
comparable across every leverage and config change in this investigation.

### PREDICTION, stated in advance

If the claim holds at this timeframe, the `<1 ATR` band should show a worse
median and more never_green. **I expect it will NOT**, for two reasons:

1. A proxy test on `range_pos_24h` pointed the OPPOSITE way. Shorts with the
   MOST room below did WORST (range_pos 0.99-1.01: +0.121, 54% win) and those
   with least did best (0.90-0.96: +0.264, 68% win). That reads as momentum,
   not room — at 0.99 the move is still extending.
2. Median hold is 1.96 minutes. A level 2% away is not an obstacle to a trade
   that lives two minutes.

Writing the prediction down first is the point. If `<1 ATR` is worse, the
claim survives a fair test; if flat, it is answered and closed.

### Limits of the proxy

The 24h extreme is a CRUDE stand-in for "next level". A volume profile would
be better and is unavailable: POC, value area and low-volume nodes need
volume-at-price, and the scanner has OHLCV only. Do not let the field name
imply more than a 24h high/low.

The rest of the article does not transfer. Its author targets one to three
trades finishing within the hour and argues explicitly AGAINST taking dozens
of small trades — the opposite of this bot's design, not a refinement of it.
It is also one session with no sample, the same limitation as CRT.

---

## THE OPEN PROBLEM: entry timing — read this first

Everything else in this document is settled or shelved. This is the live
question, and it is the operator's stated focus.

### The goal, in his words

Enter where there is **high confidence the direction is in his favour, with a
small adverse excursion** — ideally the coin consolidating before the
breakout rather than already extended. Direction first; magnitude is the
trail's job.

### Why it is THE problem

    79.1% of entries are ALREADY LOSING the first time the guardian sees them
    entered green -> n=9,  mean net ROI +3.60, win 55.6%
    entered red   -> n=34, mean net ROI -3.37, win 26.5%

and NOTHING in ~40 entry-context fields separates the two (bot/crt.py). The
screening, candidate selection and sizing are fine and the operator is
content with them. Risk sizing is verifiably excellent ($25.44 median risk on
demo, $0.45 on live, both 0.5% of equity, flat across ATR and concurrency).
The exit side has been reviewed exhaustively. **The trigger is what is left.**

### The constraint that governs everything here

**Median hold is 1.3 minutes. 80% of trades close within 3.** Any signal must
resolve inside that window to be actionable. This kills or reshapes several
otherwise reasonable ideas:

- A trigger keyed on 3m CANDLE CLOSES (CRT) works on a timescale ~2x the
  entire life of the position. Structural mismatch, not tuning.
- Forward-return scoring beyond ~5 minutes measures something the position is
  never exposed to (this invalidated the first trigger results — see below).
- "Wait one candle" is not a usable delay: the trade is usually already shut.

### What is measurable RIGHT NOW

`bot/shadow_outcomes.py` labels every REFUSED candidate — the 99% the bot
declines, which is where the sample lives — with:

    favoured_side_pct   endpoint, signed so + means the direction was right
    adverse_pct         worst move AGAINST the side (candle highs/lows)
    favourable_pct      best move in favour
    edge_ratio          favourable / adverse
    adverse_under_0.5pct   share whose adverse excursion stayed small

at 1, 2, 3, 5, 10 and 30 minutes, split by CRT and by jev verdict. The
adverse/edge columns are the direct expression of "confidence in direction
with small adverse excursion" — they did not exist until v3.74.0, and closes
alone cannot express the goal.

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/resolve_shadow_outcomes.py

### Status of the candidate triggers

- **CRT** — implemented, recorded, NEVER gating. Live: crt_agrees was False 6
  times and None once across 7 entries, never True. First scoring showed it
  mildly anti-predictive, but that was at 30 minutes — UNSCORED, not
  disproven. Timescale mismatch above is the bigger concern.
- **jev `looks_exhausted`** — a raw component on every shadow row, asks the
  same question from indicator state. Same unscored status.
- **A plain 1-2 minute delay** — the baseline both must beat. `wait_baseline`
  in the summary.

### Do not repeat these

- Do NOT cite "the better price arrives one candle later 81% of the time
  (median +0.26%)". It is computed on ENTERED trades, so it is conditioned on
  the very trigger it would be used to justify. On refused candidates the same
  statistic was a coin flip (49.5%).
- Do NOT gate on any trigger before it beats the delay baseline at a 1-3
  minute horizon on a sample well past the current 218 observations.
- Do NOT use `peak_roi` or `capture` in any of this analysis — peak is
  sampled, not tracked, and under-records by a median 9.55 ROI points on the
  moves that matter.

---

## The break-even promise depended on a poll it kept losing (v3.77.0)

The operator's rule: **once a coin touches +3% ROI it should never close at a
loss** — GUARD_BREAKEVEN_AT_ROI=3 exists to cover fees.

PHA 2026-09-21 08:29 broke it:

    08:29:24  ARMED trail placed, activation +5.00% ROI  (dormant)
    08:29:25  floor attempt 1 -> REFUSED -2021, peak +3.2%
    08:29:45  floor attempt 5 -> REFUSED
    08:30:48  roi -10.5%      <- nothing between here and the -22.5% ATR stop
    08:31:29  peak +6.6% -> floor finally placed

The floor can only REST while price is below its trigger. At +3.2% there was
about one poll to place it; by the time the order went out mark had come back
through the +2% level, so Binance correctly refused it as immediately-
triggering, and refused every retry for the same reason all the way down.

**And fail-fast makes it worse by design.** Fail-fast cuts only when
peak <= +3%, so ABOVE +3% it deliberately steps back — while the armed trail
stays dormant until +5%. A peak that touches +3.2% and reverses is exactly the
case the floor exists for, and exactly the case it cannot catch.

### Why lowering GUARD_ARM_ROI does NOT fix it

The lock is `arm_roi - callback_roi`. Setting arm_roi=3 with
GUARD_TRAIL_CALLBACK_ROI=3 locks EXACTLY 0% gross — a net loss after fees at
any leverage. Locking +2% would need callback_roi=1, which is below the
exchange minimum at 20x AND would trail every runner at peak-1%, exiting a
+100% move at +99%. That destroys the room-to-breathe the +5% arm exists for.

### The fix

`GUARD_FLOOR_TRAIL_ENABLED` places a SECOND dormant trail at entry,
activating at GUARD_BREAKEVEN_AT_ROI. Exchange-side, so no poll is in the
protection path — the same property that makes arm-at-entry work.

**Leverage decides whether the promise is keepable.** Binance rejects a
callbackRate under 0.1% of price; the rate needed is (3-2)/leverage:

    10x -> 0.100%  locks +2.0% ROI   fees ~0.9%   net +1.1%   KEPT
    20x -> 0.050% -> clamped -> locks +1.0%   fees ~1.8%   net -0.8%   NOT KEPT

At 20x GUARD_BREAKEVEN_AT_ROI must rise to ~4 for the rule to hold. The
guardian logs both warnings (cannot lock what was asked; locks less than the
round trip costs) when it places the trail.

### On stacking — the guarantee and its limit

Both trails rest DORMANT at different activation prices, so in normal
movement only one can activate, and the armed trail supersedes the floor when
it arms. **A single tick gapping from below +3% to above +5% can activate
both** before any poll intervenes — no guardian-side logic can prevent that,
because the guardian is not in the loop at activation time. Both are
reduce-only: the tighter one closes the position and the other cannot fill
against a flat position, so the sweep cancels it. The failure mode is a
redundant order, not a double close or a reversed position.

`floor_trail_id` is in the sweep's protected set. Omitting it would have
repeated the native_trail_id bug exactly — placed at entry, lands in
_all_stop_ids, swept by the next fixed-stop placement.

### First demo run, 2026-09-21 09:38-09:45 — what it showed

Placement works. SUI, METIS and EGLD all got the floor trail at entry, and
both warnings fired exactly as intended on every one:

    floor trail can only lock +1% ROI, not the +2% asked — Binance will not
    take a callback under 0.1% of price and 3%/20x needs less than that
    floor trail locks +1% ROI but a round trip costs ~1.8% ROI at 20x —
    activating it would still close NET NEGATIVE

METIS behaved correctly in the negative case: peak +2.7%, below the +3%
activation, so the floor trail never activated and fail-fast cut it at -5.3%.

**Two gaps, both introduced by v3.77.0:**

1. The floor trail was in NO `PROTECTION ... guardian holds` line. It rested
   on the exchange and was invisible. Now in the `tracked` dict as
   `floor_trail` — deliberately distinct from `floor`, which is the reactive
   profit-floor STOP_MARKET. **The naming collision is real and worth
   remembering: "floor" means two different orders.**
2. It was not cancelled at close, only swept 120s later. SUI 09:41 closed
   with BOTH the armed trail and the floor trail still resting — the armed
   trail already behaved that way and adding a third order doubled it. Close
   now cancels adaptive, armed AND floor trails.

Also seen for the first time on live: `shadow decision: rate cap (30/min)
reached`. Worth watching if candidate counts keep climbing.

### USELESS, 2026-09-21 ~10:00 — the same coin on BOTH instances

A natural experiment: both instances entered USELESS/USDT 21 seconds apart.

**The two instances do NOT see the same market.**

                sized      ATR%   recentTR    RSI   body%  lowerwick  callback
    demo 20x   0.2788     0.740      0.690   85.8    46.1       34.7     0.55%
    live 10x   0.27878    1.257      1.024   82.4    24.9       51.4     1.57%

**ATR differs by 1.70x for the same symbol at the same moment.** Demo market
data is a genuinely different feed, not just a different account. This is
direct evidence for the structural divergence AUTO_CALLBACK_ATR_MULT was
introduced to compensate for — the instances are not screening the same
market, whatever the scanner config says. Keep it in mind for ANY
cross-container comparison, not just outcomes.

**Risk sizing held exactly**, from wildly different ROIs and position sizes:

    demo  -16.6677 USDT on 5236.70 = -0.318% of wallet  (ROI -15.09%, margin $110.64)
    live   -0.2789 USDT on   90.95 = -0.307% of wallet  (ROI -12.52%, margin $2.23)

### The floor trail inverted its own ordering (fixed in v3.77.2)

On demo both trails hit -2021 on their activatePrice and retried without one.
Binance then derived its own activation from its latest price:

    armed  intended 0.2777 (+5% ROI)  -> derived 0.2771412 = +9.0% ROI
    floor  intended 0.2780 (+3% ROI)  -> derived 0.2762000 = +15.8% ROI

**The floor ended up activating LATER than the armed trail** — the exact
inversion of its only purpose. A floor without its activation is not a
degraded floor, it is a second armed trail in the wrong place.

`_create_trail_order` now takes `require_activation`, set ONLY by the floor
trail: if the activatePrice is refused it raises rather than retrying, the
floor is simply not placed, and the armed trail and fixed stop are
unaffected. The armed and rescue trails keep the retry, because for them a
derived activation is degraded rather than inverted and no trail would be
worse.

**This is why the activation path needed watching before live.** Placement had
already been seen to work; the -2021-on-activation case had not.

### v3.77.1 demo run, 2026-09-21 10:13-10:18 — both fixes verified

    PROTECTION PHA: guardian holds [adaptive=...262, armed=...298,
                                    floor_trail=...309]

Visibility fixed. Cleanup fixed too: the algo listing read 0 shortly after
the close and the orphan sweep found nothing, where the v3.77.0 session had
climbed 1 -> 4 -> 7 through the session.

The floor trail also placed WITH its activatePrice accepted this time
(0.05853, +3% ROI), so the -2021 path is intermittent rather than constant —
which is exactly why v3.77.2's refusal-to-fall-back matters: it will not fire
often, and when it does the failure was silent.

**Still unproven: a floor trail that actually ACTIVATES and FIRES.** PHA
peaked +6.7%, passing the +3% activation, but the closing order cannot be
identified from the log. Exit 0.0584101 sits below BOTH the armed trail's
implied trigger (~0.05851) and the floor trail's (~0.05848) as computed from
the POLLED low — which only means the exchange saw a lower low than the
guardian sampled. Same peak_roi sampling problem, not a new defect.

Settle it with `fetch_my_trades` on a closed position and match the closing
order id against the ids in the log. Until that is done, the activation path
has never been observed end to end.

### Before enabling on live

Run it on demo and confirm from the logs that every resting order is
accounted for after each close. Demo at 20x is also the worst case for
gap-throughs, since the ROI thresholds sit closer together in price terms —
and it will show the +1% lock rather than +2%, which is worth seeing.

---

## A missing break-even floor was invisible on the dashboard (v3.76.1)

PHA 2026-09-21 08:29. The profit floor was refused by the exchange:

    PROTECTION-NO-FLOOR PHA: peak reached +3.2% but NO profit floor is
    resting (attempt 1, wanted +2.0% ROI, currently +3.2%)
    exchange said: -2021: Order would immediately trigger.

and again at attempt 5. The floor was only placed at 08:31:29, so the
position ran ~2 minutes with break-even unlocked. **The dashboard showed
nothing**, because `snapshot()` exposes only `unprotected_reason` and
`_floor_unavailable` never set it — it logged at ERROR and called `_record`.

The -2021 itself is a race, not a bug: the floor level is derived from a peak
observed up to a poll ago, and by the time the order is sent price has come
back through it, so a buy-stop for a short would trigger instantly. Expected
on a 2.5s poll against a fast-moving mark; it retries and eventually places.

### The fix, and why it is a NEW field

`GuardState.floor_missing_reason`, set by `_floor_unavailable` and cleared
when the floor lands. Deliberately NOT `unprotected_reason`: PHA still had a
fixed stop and two trails resting. Reusing that field would have shown
"UNPROTECTED" for a position that was protected — just not at break-even —
and that banner tells the operator to close the position or accept the risk.

The dashboard now renders a distinct amber notice, weaker than the red
unprotected banner, saying the stop and trails are resting and only profit
lock is missing. `floor_attempts` is exposed alongside it.

Note the module's own docstring already recorded the earlier version of this
bug: the dashboard kept showing "Stop @ ROI +2%" because that field is the
guardian's INTENT, not what the exchange holds. The logging was fixed then;
the dashboard was not.

---

## Shadow backoff on provider outages (v3.76.0)

2026-09-21 07:23-07:27 TypeSafe degraded and recovered on its own:

    503 no healthy upstream            (request id ABSENT — died at their edge)
    529 high traffic, try again later  (request id PRESENT — shed deliberately)
    ReadTimeout (timeout=10.0)
    200 in ~660ms                      recovered

**Not a proxy problem, and the proxy was never involved.** The SDK runs on
`httpx2`, which reads `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` — not
`SOCKS_PROXY`, which is what compose sets and only ccxt consumes explicitly.
A clean, service-specific 503 in ~310ms is an upstream answering promptly; a
broken proxy gives connection refused, timeouts or 407.

Throughout, every candidate kept firing into a service explicitly asking to
be left alone, writing one UNKNOWN row and one WARNING each.

`BACKOFF_AFTER=5` consecutive SERVER-side failures now pauses judgements,
starting at 30s and doubling to a 300s ceiling. One good response clears both
the pause and the escalation. Replayed against the real sequence: 30
candidates offered, **5 calls made, 25 skipped**.

Deliberate boundaries:

- **4xx never backs off.** A 400 is a config error waiting cannot fix, and
  `Unknown model` still latches permanently. Transience is matched on the
  message because the SDK maps 503 and 529 onto the same exception class.
- **In-flight threads cannot extend an active pause** — threads dispatched
  before the pause land after it, and letting each re-arm would stretch 30s
  indefinitely.
- After the first `BACKOFF_AFTER` failures the per-candidate line drops to
  DEBUG. The outage buried everything else in the log.

This is politeness and log noise, **not data integrity** — the resolver
already drops UNKNOWN rows and the shadow path is advisory. A test pins that
`decide_async` still returns None for every candidate during an outage.

---

## The horizons were 23x too long (v3.75.0)

Hold time, measured over 771 trades:

    p50   1.3 min          3 min:  80% closed
    p75   2.3              6 min:  92%
    p90   4.9             15 min:  98%
    p95   7.6             30 min:  99%
    p99  17.3             60 min: 100%

`HORIZONS_MIN` was `(3, 6, 15, 30, 60)` and `summarize()` defaulted to 30.
**Every trigger conclusion in SETTLED #7 was therefore scored on a window 23x
longer than the median trade** — it measured whether CRT and jev predict
something the position is never exposed to. Treat #7 as unscored, not as a
negative result.

Now `(1, 2, 3, 5, 10, 30)`, default horizon 2. One minute is the floor:
`fetch_ohlcv` has no finer candle.

**A sharper consequence for CRT.** The median trade closes in 1.3 minutes —
BEFORE the 3m scanner candle it entered on has closed. A trigger keyed on
candle closes is operating on a timescale roughly twice the entire life of
the position. That is a structural mismatch, not a tuning problem, and it
should be settled before more work goes into CRT.

The delay baseline moved with it: `wait_baseline` is now `one_minute` /
`two_minutes` rather than 3m candles. A trade that waits one 3m candle has
usually already been closed.

Because median hold sits inside the first candle, the FIRST candle's high and
low carry most of what a position lives through — which is why the v3.74.0
excursion data matters more here than closes do.

### Two things that are NOT problems (checked, 2026-09-21)

**Live's tiny position sizes are correct.** 157 live trades risk a median
$0.45 with p10-p90 of $0.43-$0.47 — 0.5% of an ~$89 wallet, held to four
cents, a TIGHTER distribution than demo's. Live is a data-collection account
at this size; do not read its net P&L as evidence about the strategy. Demo
(implied wallet $5,156) is where P&L means something.

**Concurrency limits are not needed yet.** `ENTRY_MAX_POSITIONS=6` but
concurrency is 1 in 593 cases, 2 in 126, 3 in 48, and never higher. Sizing
holds flat across all three ($86-$89 margin, $25.3-$25.7 risk). Correlated-
exposure caps would solve a problem that does not exist. Directional
concentration also does not hurt: all-same-side n=139 median net +$0.74 (win
57%) vs mixed n=35 median -$0.22 (win 46%) — the opposite of a warning,
consistent with the strategy working when there is a real market-wide move.

**No BNB fee discount, by choice** — a constant fee rate is worth more than a
discount that complicates cost basis mid-history. Do not propose it again.

---

## Path capture: adverse and favourable excursion (v3.74.0)

The entry-timing goal is "high confidence in direction AND a small adverse
excursion — ideally consolidating before the move". That is a statement about
the PATH, and the resolver only recorded ENDPOINTS: a candidate that ran -8%
before +12% scored identically to one that went straight to +12%.

Every observation now carries, per horizon:

    adverse_pct      worst move AGAINST the side, clamped at 0
    favourable_pct   best move IN FAVOUR
    edge_ratio       favourable / adverse; None (never inf) when adverse is 0

Computed from candle HIGHS and LOWS, not closes — the point is what the
position would have LIVED THROUGH, and a close hides the wick that would have
taken out a stop.

`summarize()` reports `path`, `path_by_crt_agrees` and `path_by_verdict`,
each with median adverse, median favourable, median edge ratio, and
`adverse_under_0.5pct` (the share of observations whose adverse excursion
stayed under 0.5% of PRICE — stated in price because ROI depends on leverage).

**This also gives the triggers a fairer test than they have had.** CRT and
`looks_exhausted` were scored on 30-minute endpoints (SETTLED #7) and both
looked mildly anti-predictive. If CRT's real claim is "entry here has a small
adverse excursion", endpoint returns were never the right scorer for it.

Sample discipline unchanged: 218 observations, 20 in the jev-ENTER class, 40
CRT-True. Richer analysis on a small sample is how patterns that are not
there get found. Let it accumulate a week before fitting anything.

### A recurring error worth naming

Three times in this investigation a quantity denominated in DOLLARS was
reasoned about in ROI%:

- "Is ATR_STOP_MAX_ROI=30 too permissive?" — no. `ENTRY_RISK_PCT=0.5` is the
  bound; `ATR_STOP_MAX_ROI` sizes MARGIN DOWN so the dollar risk is constant.
  Verified across 762 trades: risk at stop is $25.2-$25.7 in every ATR band
  from 0 to 2.5% while stop_roi runs 16 -> 30 and margin runs $149 -> $85.
  Median $25.44 = 0.5% of a $5,000 wallet.
- "The cap isn't a cap" (fail-fast overshooting -5%) — wrong. The exchange
  ATR stop is the bound; fail-fast pulls losses IN from it. Every fail-fast
  exit closed well inside its own stop (SAGA -30.0 -> -10.91, NEAR -25.2 ->
  -6.13).
- Give-back and capture in ROI points across containers running 10x and 20x.

**ROI% is not comparable across leverage or across position sizes. Convert to
price % or to dollars before comparing anything.**

### One genuine finding from that thread

    ATR band     n    stop_roi   margin$   risk$
    2.5%+       52      30.0        1.6     0.49

Above ~2.5% ATR the sizing hits `ATR_STOP_MAX_ROI=30`, cannot widen further,
and compensates by collapsing margin to $1.60 — risking 49 cents. Those 52
trades (7% of entries) are economically meaningless: fees likely exceed the
risk, while still consuming a position slot, a symbol cooldown and scanner
attention. `AUTO_MAX_ATR_PCT` exists for this and is 0 (off). 2.5% is where
the data says sizing stops working. NOTE it would cut entry rate, which is
the variable currently under experiment.

---

## REJECTED algo orders — the alarming reading was WRONG (v3.78.1)

**READ THE CORRECTION AT THE END OF THIS SECTION BEFORE ACTING ON IT.** The
original conclusion — that the guardian holds refused orders as resting
through the life of a position — is NOT supported by later evidence.

PHA demo 2026-09-21 10:18. The guardian logged `FLOOR TRAIL resting
id=...309`, `TRAIL-RESPONSE` showed the activatePrice accepted and kept, and
every `PROTECTION` line listed it as held. The algo HISTORY
(`fapiPrivateGetAllAlgoOrders`) says:

    armed         1000000212987298  FINISHED   <- actually closed the position
    FLOOR         1000000212987309  REJECTED   <- never rested
    adaptive      1000000212987262  CANCELED
    fixed         1000000212987373  CANCELED
    profit-floor  1000000212987532  REJECTED   <- also refused

**Two of five protective orders were refused and the guardian believed all
five were resting.** The 08:29 PHA trade shows the same shape: armed
...852893 REJECTED while the profit floor ...855212 FINISHED.

The module already knew this was possible — "the algo endpoint returns 200
with an id and refuses the order" — and `_accepted_id` exists to catch it. It
cannot: **the rejection is ASYNCHRONOUS.** The POST returns 200 with a real
id and the status flips afterwards.

### Why polling is legitimate here, unlike for price

Polling for PRICE loses information permanently — a peak not sampled is gone,
which is why peak_roi under-records by a median 9.55 ROI points. Polling for
ORDER STATUS loses nothing: REJECTED is durable and discrete, so a later read
returns the same answer.

What remains is a BLIND WINDOW between placement and verification — one
guardian cycle (~2.5s), against the entire life of the position before.

`_verify_protection` runs ONCE per position, one cycle after placement, reads
the algo history, and DROPS any tracked id whose status is REJECTED /
EXPIRED / CANCELED. Deliberately conservative: an unreadable algo book or an
id absent from the history (the history has a retention window) changes
nothing — silence beats dropping live protection on a guess.

### It DETECTS, it cannot PREVENT

Likely cause of the rejections, still unconfirmed: all three trails are
`reduceOnly` at FULL position quantity, so adaptive + armed + floor is 3x the
position in reduce-only orders, and Binance refuses reduce-only beyond
position size. Two fitted, the third did not.

**If that is the cause, the floor trail can never rest alongside the other
two, and the design is wrong rather than the implementation.** The fix would
be fewer trails, not better checking.

### CORRECTION, same day, v3.77.2 demo run

Two positions (MUBARAK, PHA) with all three trails. Every one came back
**CANCELED**, none REJECTED:

    MUBARAK  FLOOR ...104 CANCELED   armed ...092 CANCELED   adaptive ...044 CANCELED
    PHA      FLOOR ...733 CANCELED   armed ...706 CANCELED   adaptive ...653 CANCELED

**The reduce-only aggregate theory is DEAD** — three reduce-only trails at
full position size coexist without trouble.

And the REJECTED pattern has a simpler explanation:

    v3.77.0, PHA 10:18  guardian did NOT cancel armed/floor_trail at close
                        -> left resting, position vanished underneath them
                        -> REJECTED
    v3.77.2, both       guardian DID cancel all three at close  -> CANCELED

**REJECTED almost certainly means "invalidated when the position closed", not
"refused at placement".** The profit floor fits: placed 10:18:20, position
gone by 10:18:23, never cancelled by that build. The one order showing
FINISHED is the armed trail — the one that actually fired.

So the orders WERE resting during the position's life. This was a bookkeeping
artefact of the pre-v3.77.1 cleanup gap, not a protection failure.

### What still stands

`_verify_protection` (v3.78.0) is still worth having: a genuine
placement-refusal is possible, the module's own comment says the algo
endpoint can return 200 and refuse, and the check costs one request per
position. But it is a SAFETY NET, not a fix for an active defect — and its
warning text should not be read as evidence that one occurred.

**Do not repeat this inference.** A terminal algo status read AFTER a close
cannot distinguish "refused at placement" from "invalidated by the close".
Only a status read WHILE the position is open can, which is exactly what
_verify_protection does.

### Consequences

- **Do NOT enable GUARD_FLOOR_TRAIL_ENABLED on live yet.** Not because of
  rejections — that reading was wrong — but because it has still never been
  observed ACTIVATING and FIRING. Placement, coexistence and cleanup are all
  now confirmed on demo.
- Historical `PROTECTION ... guardian holds [...]` lines are NOT known to be
  wrong. That claim was withdrawn.

---

## peak_roi is SAMPLED, not tracked — do not trust it (v3.73.6)

**The single most important instrumentation finding in this investigation.**

Across 770 pooled trades, `final_roi` EXCEEDS the recorded `peak_roi` in 73 of
them (9%; 66 of 443 TRAIL exits, 15%). That is arithmetically impossible if
the peak were tracked — final cannot beat a maximum that was actually
observed.

    LSK    peak  23.57   final 118.40   MISSED 94.83 pts   lag 2.95s
    LSK    peak  32.63   final 124.80   MISSED 92.17 pts   lag 1.80s
    SYN    peak  30.88   final  96.45   MISSED 65.57 pts   lag 1.55s
    CELR   peak   3.30   final  56.89   MISSED 53.59 pts
    SAGA   peak  -5.72   final  39.01   MISSED 44.73 pts   <- NEGATIVE peak

    median under-recording among the 73: 9.55 ROI pts, max 94.83

The guardian polls every ~2.5s; the exchange trail tracks tick-by-tick. On a
move that runs 100% ROI in minutes the poller never samples the top.

### The same defect from the other side

`peak_roi` is MONOTONIC, so ONE bad reading is permanent in either direction.
The operator reports a flash of overwhelmingly positive unrealised PnL right
after a fill, visible in Binance's own UI, before it settles. A phantom high
pins the peak for the life of the position; a missed run never enters it.
32 trades have `peak_roi` exactly equal to `roi_at_first_sight`, never
revisited.

### What this compromises

Everything keyed on `peak_roi`: `is_armed`, `desired_stop_roi`, the breakeven
trigger, fail-fast. A position can sit at +40% on the exchange while the
guardian believes +5% and has not armed.

**This is the argument for arm-at-entry**, and it is the operator's: an
exchange-placed trail activates and trails on the venue's own ticks, immune to
the poll. Placement at entry is not an optimisation, it is the only mechanism
that does not inherit this defect.

### What it invalidates in this document

- SETTLED #5's "capture 59% both sides" used `final/peak` — **INVALID**, peak
  is understated and worse on fast moves, which inflates capture.
- Any give-back computed as `peak_roi - final_roi` is UNDERSTATED.
- SETTLED #6 STANDS: its tables use realised `final_roi` and ATR bands, not
  peak. (Rows were filtered on `peak_roi > 0`, a mild selection effect only.)

### Reconstructing the true peak

For a trail exit the callback is known, so the real peak is recoverable:

    implied_peak = final_roi + callback_pct * leverage

Validated on the 2026-09-20 demo winners, where the log states the callback:

    trade        cb%   recorded peak   IMPLIED peak   under-record
    S/USDT      0.73       100.2%         110.0%         9.8 pts
    KMNO 23:46  0.58        94.8%         103.5%         8.7 pts
    KMNO 22:38  0.38        33.4%          38.9%         5.5 pts

Use this rather than `peak_roi` for any give-back work.

---

## SETTLED — conclusions reached and reversed, do not re-litigate (v3.73.3)

Several lines of reasoning in this investigation were pursued, tested against
data, and ABANDONED. They are recorded here with what killed them, because
each one looks plausible on a fresh read and costs hours to re-derive.

### 1. "MARK_PRICE vs last price is costing us money" — NOT SUPPORTED

Pursued hard after STG (peak +23.2%, closed -8.08%). Killed by the data: over
10 closed trades the observed-vs-final gap is two-sided and median ZERO
(SAGA gained +7.00 from divergence, STG lost 26.69). Demo diverges twice as
much as live (median |div| 0.151% vs 0.068%, n=80), so demo OVERSTATES it.
One outlier cannot carry the case. The tail is real but rare (AKE 2.377% =
47 ROI points at 20x) and remains UNADDRESSED by choice.

**Do not switch workingType without many more trail exits.**

### 2. "The armed trail should arm LATER on volatile coins" — WRONG, and my
### own addition, not part of the real fix

v3.72.0 correctly widened the callback (0.15% was 0.25x STG's ATR). It ALSO
deferred arming, via `effective_arm_roi`. The operator challenged this and was
right. On the live config the deferral costs 1-3 ROI points of locked stop
between +6% and +8.3% and buys nothing above +9%.

The thing deferral was meant to prevent — a trail engaging below entry — is
not a risk here: `GUARD_BREAKEVEN_AT_ROI=3` / `GUARD_BREAKEVEN_STOP_ROI=2`
puts a stop at +2% once peak hits +3%, above where a wide trail would engage,
and the tighter order fires first.

**Set `GUARD_MIN_TRAIL_LOCK_ROI=0` if the floor is ever enabled on live.**

### 3. "Build a conditional deferral fix" — PROPOSED TWICE, DO NOT BUILD

The idea: arm at `cfg.arm_roi` when a profit-locking stop already exists. It
was justified by a table showing the fixed-stop RATCHET being suspended during
deferral. **That table was read from `desired_stop_roi()`, a pure function
whose armed branch is DEAD CODE in this deployment.**

`GUARD_RATCHET_ENABLED` defaults to false and neither compose file sets it.
Once a native trail rests alongside a fixed stop — both containers, via
`GUARD_ARM_AT_ENTRY=true` — repositioning is suppressed entirely. There is no
ratchet to lose.

With the ratchet off, arming gates ONE thing: whether the native trail is
live. Between +5% and the deferred level an early-armed trail's give-back
(7.4% ROI at 0.75x) sits BELOW breakeven's +2%, so breakeven binds either way
and **the exit price is identical**. The fix would change when an order is
placed, not where the position gets out.

### 4. "Chain GUARD_TRAIL_CALLBACK_ATR_MULT to AUTO_CALLBACK_ATR_MULT" —
### MISTAKE, made in v3.72.1

Done in response to a correct observation (the two should not drift apart).
Wrong here specifically: `AUTO_CALLBACK_ATR_MULT` is the INDEPENDENT VARIABLE
of a running experiment with a pre-committed control, so chaining silently
coupled an exit-side setting to it and gave the comparison three differences
instead of one.

The two multipliers answer different questions. The entry callback is a
per-instance correction for an entry-RATE divergence between deployments. The
guard floor asks how much room a running position needs against the coin's own
movement — a property of the INSTRUMENT, identical across containers that
screen the same universe. Revert the chained default to a literal and pin both
containers explicitly.

### 5. "The v3.72 floor improves outcomes" — NOT SHOWN

Demo ran it from ~17:50 on 2026-09-20. The deferral works mechanically (9/9
armed trades compliant after deploy; UB 18:29 armed at peak 12.9 against a
12.0 needs-level; VVV 02:32 correctly NOT armed at peak 17.4 against 21.4).
But normalised for leverage:

    DEMO after floor (0.75x ATR)   give-back 0.466% of price   capture 59% of peak
    LIVE (floor off, 0.30x ATR)    give-back 0.301% of price   capture 59% of peak

**Identical capture.** The wider trail gives back more price for the same
fraction of peak. Demo's better absolute numbers are 20x vs 10x on different
coins on one day. n=11 vs 12. The floor's justification remains "0.30x ATR is
inside the candle and STG failed at 0.25x" — a structural argument, NOT an
outcome result.

### 6. "The armed callback must not sit inside the candle" — NOT SUPPORTED
###    (this was the LAST surviving justification for v3.72.0; it failed)

Tested on 395 trail exits pooled from 726 historical trades (512 demo + 214
live, 09-11 to 09-20). The claim predicted tight callbacks would do worse.
They do better, monotonically:

    cb/ATR        n    p10     p25   median   worst    <0
    0.00-0.15   157   -1.36    5.53   14.84  -47.07   11%
    0.15-0.25   119   -0.62    2.80    7.15  -10.17   10%
    0.25-0.40    92   -1.43    2.85    6.77  -10.90   17%
    0.40+        27   -5.44   -1.86    1.78   -7.64   26%

`cb/ATR` is largely leverage, so that table alone is demo-vs-live. The clean
control is SAME ATR band, DIFFERENT callback width (20x = 0.15% of price,
10x = 0.30%), final in price terms:

    ATR band   20x n  20x final%px   10x n  10x final%px
    0.0-0.6       76      0.371          9      0.144
    0.6-0.9      100      0.371         26      0.270
    0.9-1.5       66      0.560         11      0.316
    1.5+          91      0.845         15      0.738

The tighter callback wins in all four bands.

**The tail does not rescue it either**, which is where the argument lived. The
single worst outcome IS in the tightest band (-47.07), but the loss RATE is
flat at 10-11% for tight callbacks against 26% for wide, and p10 is no worse.
One bad trade does not outweigh a monotonic median across four bands.

The mechanism story was not wrong — a callback inside typical movement will
sometimes fire on an ordinary candle, and STG did. What was never checked is
the other side: a tight trail also exits FAST when a move genuinely reverses,
and across 395 exits that is worth more than the false triggers cost.

**v3.73.4 therefore defaults `GUARD_TRAIL_CALLBACK_ATR_MULT` to 0** and
removes the chaining to `AUTO_CALLBACK_ATR_MULT`. Both containers can drop the
variable entirely; neither needs it set. `GUARD_MIN_TRAIL_LOCK_ROI` becomes
inert (with no floor, `effective_arm_roi` resolves to `cfg.arm_roi`).

Caveats, so this is not read as the REVERSE claim: `cb/ATR` is mostly
leverage, so tight-vs-wide is largely demo-vs-live with all the instance
differences that carries; high-ATR coins produce bigger moves regardless of
trail width, which flatters the tight band; and `capture` (final/peak) is a
BIASED metric that should not be used — a tight trail truncates the peak it is
measured against. The honest position is "no support for widening", not
"tightening is proven".

### 7. First real trigger results — NEITHER CRT NOR jev SEPARATES ANYTHING

Live, 2026-09-21. 2064 decision rows collapsing to 218 independent
observations, all `bot_decision=SKIP`, forward returns at 30m:

    jev ENTER   n= 20   +0.2225%      jev SKIP    n=198   +0.3335%
    CRT True    n= 40   +0.2370%      CRT False   n=178   +0.3335%

**Each trigger's "yes" class did WORSE than its "no" class**, by ~0.10pp. Two
independent hypotheses, both pointing the wrong way. Positive classes are
small (20 and 40) so this is not significance — but it is not encouragement
either, and gating on either would currently be gating on noise at best.

Read these RELATIVELY. Every cell is positive because the whole population
drifts favourably: the scanner selects momentum and momentum continues on
median. The +0.31% baseline is NOT "refusing cost money" — there is no stop,
no fee and no slippage in it.

**The one-candle wait baseline is a coin flip:** median 0.0%, better 49.5% of
the time (two candles: +0.0986%, 54.6%).

This CONTRADICTS the `wait_*` figure (better price one candle later 81% of the
time, median +0.26%) used earlier to argue for a structural trigger. The two
populations differ: `wait_*` is computed on ENTERED trades, this on REFUSED
candidates. That is exactly the problem — the 81% figure is conditioned on a
population selected by the entry trigger, so it cannot justify changing that
trigger. **Do not cite the 81% number in trigger arguments again.**

`decision_rows_covered: 2064` vs `observations: 218` — the collapse is doing
real work; a naive row count overstates the sample ~9.5x.

Next read is dispersion, not another median: a 0.10pp gap means nothing
without the spread.

### 8. The volatility floor MEASURABLY COST the two biggest winners

The clearest single test of v3.72.0's floor, on the trades the account
actually depends on. Demo 2026-09-20, floor ON, callbacks confirmed in the log
(`armed callback raised 0.15% -> 0.73%` etc.):

    trade        cb%    implied peak   actual final   with 0.15% cb   floor COST
    S/USDT      0.73       110.0%          94.7%         106.8%        12.1 pts
    KMNO 23:46  0.58       103.5%          91.4%         100.4%         9.0 pts
    KMNO 22:38  0.38        38.9%          31.2%          35.9%         4.7 pts

Deferral was irrelevant to these — their deferred arm levels were +16.6% and
+9.6% against peaks of 110% and 103%, so they armed many times over. The floor
reached them purely as callback WIDTH, and a 15% ROI give-back on a 110% peak
is a 14% tax on the best trade of the day.

The only surviving counter-argument — that a 0.15% callback might have fired
early and never reached 110% — is exactly what SETTLED #6 tested across 395
historical exits at that callback. Tight callbacks produced BETTER medians,
not truncated winners.

**This matters most under the operator's stated mechanism**: the account is
carried by capped downside plus two or three large winners. A wider trail is a
flat tax on precisely those winners.

### What actually survives from all of this

- (The armed-callback-width claim did NOT survive — see #6 above. v3.72.0's
  floor is now off by default and nothing from it remains active.)
- `confidence` and the shadow model name were genuinely broken (v3.69-3.70).
- Shadow rows were not independent (v3.71).
- Arm-at-entry was broken by v3.72.0 and fixed in v3.73.1.
- The supersede sweep had never protected `native_trail_id` — see below.

---

## DEMO IS LIVE DATA WITH A ~0.75 FACTOR ON ATR — read before porting anything

**Demo and live are SIMILAR, not EQUIVALENT.** The price path matches; the
amplitude does not.

### The measurement

Same symbol, paired within 30 minutes, 13 pairs from the 2026-09-21 exports:

    ATR       demo/live   median 0.751   mean 0.769   sd 0.137  (range 0.66-1.21)
    advance   demo/live   median 0.937
    RSI, 24h change                      agree to ~1 point

USELESS, 21 seconds apart, is the cleanest single pair:

                sized      ATR%   recentTR    RSI   body%  lowerwick
    demo 20x   0.2788     0.740      0.690   85.8    46.1       34.7
    live 10x   0.27878    1.257      1.024   82.4    24.9       51.4

**Candidate selection agrees, chart lines are similar, RSI and 24h change
match — only intra-candle RANGE is compressed.** Demo has the same trend with
~25% less noise (advance 0.937 / ATR 0.751 = 1.25x better trend-per-noise).

Supporting: the same scan pass yields 21 of 40 movers on demo vs 10 of 40 on
live. ATR is NOT the scan gate (min_atr is an auto-trade refusal, applied
later), so that gap comes from cleaner RSI/EMA states — consistent with less
noise.

### EVERY ATR-derived setting means something different on each instance

    AUTO_CALLBACK_ATR_MULT       known, deliberately different
    ATR_STOP_MULT=1.5            demo stops ~0.75x as wide in real terms
    AUTO_MIN_ATR_PCT=0.5         demo needs ~0.67% live-equivalent to pass
    GUARD_TRAIL_CALLBACK_ATR_MULT  same (currently 0)

**Conversion: a demo mult of M is equivalent to a live mult of 0.751 x M.**
So demo 0.75 == live 0.563, NOT live 1.25.

### The performance gap, and why LEVERAGE EXPLAINS NONE OF IT

2026-09-21, 43 demo trades vs 37 live:

                          demo      live
    win rate              58%       59%      <- essentially identical
    gross P&L/trade      0.128%    0.037%    of wallet
    fees/trade           0.040%    0.036%    of wallet — proportionally EQUAL
    fees as % of gross    31.5%     98.4%

**Fees scale with notional, and notional scales with leverage, so leverage
multiplies gross AND fees equally.** The leverage-free measure is how many
times its own fee a trade captures:

    demo 3.18x     live 1.02x     ratio 3.13x

Two measured effects roughly account for it:

    trend/noise advantage on demo                         1.25x
    live waits a 2.22x DEEPER retracement before filling
      (live 1.25 x ATR_live = 1.250 vs demo 0.75 x ATR_demo = 0.563)
                                                 1.25 x 2.22 = 2.78x vs 3.13x

The second is the actionable half. **CORRECTED 2026-09-21 by the 11:49
exports** — see below; per-trade PRICE outcomes do NOT show the multiplier
costing edge, and the claim that it does is withdrawn.

**Do NOT try to close the gap with leverage.** It cancels.

### How to work with this

Demo REMAINS the test case — it has far more data, and the operator's
decision. The requirement is that the 0.751 factor is applied when a
parameter is ported:

    live_equivalent_mult = 0.751 x demo_mult

**Most development to date was optimised ON DEMO**, which has 1.25x better
trend-per-noise. Every parameter tuned there is tuned for an easier market,
and the bias is SYSTEMATIC, not random. That is the single most important
consequence of this section.

### Caveats

43 and 37 trades, one day. The 0.751 ATR factor is solid (n=13, tight spread).
The edge decomposition is a HYPOTHESIS — 2.78 vs 3.13 is close enough to be
coincidence, and correlation is not causation. Re-measure before treating the
callback-depth effect as established.

---

## The multiplier's measured effect — and what it does NOT show (v3.79.0)

From the 11:49 exports, 38 live (1.25) and 45 demo (0.75), **all in PRICE %
with leverage divided out**, atr_floor trades only:

                     n   drift    first_sight   underwater
    live 1.25       37  -0.669%     -0.176%        95%
    demo 0.75       36  -0.270%     +0.023%        39%

    FINAL         median    mean   trimmed      p10      p90   win
    live 1.25     +0.157  +0.170   +0.124   -1.266  +1.461   59%
    demo 0.75     +0.081  +0.235   +0.180   -1.082  +1.274   53%

**The multiplier does exactly what it should on FILL QUALITY**: the wider
callback fills 0.669% better than its sized price against 0.270%, 2.5x the
improvement. And it is underwater at first sight 95% of the time against 39%.

**Both are real and they are not contradictory.** A better price comes from
waiting for a bigger bounce, and you enter while that bounce is still
running, so it continues briefly.

**On OUTCOME they are tied.** Live has the better median and win rate, demo
the better mean, and trimming one outlier from each tail flips the mean. At
n=37/36 that is a coin toss.

### The multiplier change had a SECOND effect nobody asked for

The callback is `max(ratio-derived, ATR floor)`. Lowering the floor does not
just narrow the callback — it hands trades to `AUTO_CALLBACK_RATIO` instead.
Live, around the 2026-09-21 12:34 UTC change:

    BEFORE (1.25)   n=38    atr_floor 97%
    AFTER  (0.563)  n=23    atr_floor 52%   ratio 48%

Eleven trades left the experiment entirely, because the tool grouped only
`atr_floor` rows. The comparison showed n=12 while live had taken 23. A result
read as "0.563 vs 1.25" would partly have been "ratio-source vs floor-source",
and the ratio callbacks are narrower again (0.420% vs 0.510% median).

`compare_callback_mult.py` now shows every non-floor source as its OWN
cohort, labelled by source name, never merged and never given a numeric
multiplier.

### That cohort is ATR-SELECTED — do not read it as a result

    cohort    n   ATR med   final median   win   x its fee   never_green
    0.60     12    0.900      +0.176       67%    -0.10x         8%
    1.20     37    1.012      +0.157       59%    +1.32x        30%
    ratio    11    0.556      +0.243       82%    +2.29x         0%

The ratio cohort looks best on every measure. **It is also trading coins with
roughly half the ATR**, and that is not a coincidence: the floor binds when
ATR is HIGH relative to the distance, so "ratio" is largely a label for CALM
COINS. Membership depends on ATR and ATR predicts outcome — the split is
ENDOGENOUS.

The tool now prints `!! COHORT 'x' IS ATR-SELECTED` when a non-floor cohort's
median ATR is more than 25% away from the floor groups'.

**Practical consequence for the experiment**: at 0.563 the floor governs only
about half of live's trades, so the comparison accumulates at half the rate
and the two halves are not exchangeable. Either accept the slower read, raise
the floor to ~0.8-0.9 so it binds again (abandoning the demo-parity value),
or lower `AUTO_CALLBACK_RATIO` so the floor keeps winning (which changes a
second variable).

### FIRST FULL READ of the multiplier, 2026-09-22 (n=34 vs 37)

    mult    n    ATR   median    mean  trimmed     p90     p10  win  nevergrn  x fee
    0.60   34  1.290   +0.169  -0.035   +0.007  +0.834  -1.064  62%      15%  -0.08
    1.20   37  1.012   +0.157  +0.170   +0.124  +1.461  -1.266  59%      30%  +1.32

**0.60 wins on median, win rate and never_green. It LOSES on everything that
carries the account.** Mean is negative, trimmed mean is near zero, and p90
collapses from +1.461% to +0.834% of price — the narrower callback caps the
winners. Economics is the plainest statement: 0.60 captures -0.08x its own
fee and is down 1.07 USDT over 34 trades, against 1.32x and +0.37.

Under the operator's stated mechanism — capped downside plus two or three
large winners carrying the account — cutting p90 by 43% is precisely the wrong
trade, however good the median and win rate look.

**CONFOUNDED, and the tool failed to say so.** Median ATR 1.290 vs 1.012 =
1.27x, outside the 1.25 tolerance. The regime check did not fire because it
was guarded with `if len(keys) == 2` and a third ('ratio') cohort had
appeared. Fixed in v3.79.3 to compare floor groups pairwise regardless of how
many cohorts exist.

The confound makes the result STRONGER, not weaker: 0.60 traded MORE volatile
coins and still produced smaller winners.

**Reading**: 0.60 is not supported. It trades fewer bad entries for
materially smaller good ones. If a value is to be tested against 1.25, the
design-intent argument favours 0.75 (0.75x of live's OWN noise), not 0.563 —
see the entry above. Note also that at 0.60 the floor governed only 76% of
trades, so the cohorts are not fully exchangeable.

### never_green is the sharpest split in the data

`never_green` = the position never traded above entry (`peak_roi <= 0`). From
the same exports, atr_floor trades only, price %:

                  n   never_green   their final   others final
    live 1.25    37       30%         -0.574%       +0.238%
    demo 0.75    39       23%         -0.853%       +0.355%

**Nothing else in this dataset separates winners from losers that cleanly.**
30% of live trades never trade above entry and lose ~0.57% of price; the
other 70% make ~+0.24%.

It is plausibly connected to the multiplier: a WIDER callback fills deeper
into the bounce, so the position starts further underwater (first_sight
-0.176% live vs +0.023% demo) and needs more recovery just to reach entry.
Lowering the multiplier should REDUCE the rate. **That is the sharpest
prediction of the live change**, and `compare_callback_mult.py` now reports
it per group.

CAVEAT: `peak_roi` is SAMPLED and under-records by a median 9.55 ROI points
on fast moves, so a position that went green between polls records
`peak <= 0` and is counted. This OVER-counts never_green, and does so more
when moves are fast. Treat the RATE as comparable between groups only if
their observation lags are similar.

Note also that `dead_on_arrival` (peak <= 0 AND <= -20% ROI) fires far less on
live: `DEAD_LOSS_ROI=20` is denominated in ROI, so it is -1.0% of price at 20x
but -2.0% at 10x. Live had ZERO in this window against demo's four. Its
absence on live is uninformative, not reassuring.

### Two claims WITHDRAWN

1. "The callback-depth effect costs edge." NOT SUPPORTED by per-trade price
   outcomes. The earlier 3.13x decomposition used dashboard WALLET
   percentages, which fold in position sizing; price move per trade is the
   cleaner measure and it shows no gap.
2. The 95%-underwater figure has been treated as an entry-quality problem.
   Much of it is an ARTEFACT OF THE WIDER CALLBACK, not a signal about the
   trigger. `roi_at_first_sight` is not comparable across different
   multipliers.

### The only clean test, and how to run it

Cross-container comparison carries the 0.751 ATR factor, a leverage
difference and a different symbol set — confounds larger than the effect.
Split WITHIN one instance instead:

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/compare_callback_mult.py

Groups identify THEMSELVES from `callback_pct / atr_pct` per trade, so there
is no deploy timestamp to remember and a re-run months later still splits
correctly. Only `callback_source == "atr_floor"` trades are grouped — the
first version invented cohorts at 0.70/0.90/1.00 that were ratio-sourced
trades with coincidental ATRs.

Everything is PRICE %, never ROI. Reasoning in ROI about a price-denominated
quantity caused four separate wrong conclusions in this investigation.

The tool refuses to declare a winner. It flags groups under 30 trades, a
leverage difference between groups (no longer within-instance), and a median
ATR difference over 25% (the groups did not trade the same market). At ~40
trades/day that is roughly a week per group.

**Run it BEFORE deploying the change** — with one multiplier present it
prints the BEFORE baseline and says so.

---

## AUTO_CALLBACK_ATR_MULT: why live runs 1.25 and demo runs 0.75

**This asymmetry is deliberate and under test. Do not "fix" it.**

### What the setting does

The callback is the retracement a trailing ENTRY order waits for before it
fills. `AUTO_CALLBACK_ATR_MULT` floors that callback at a multiple of the
coin's ATR. It governed 38 of 43 live entries (`callback_source=atr_floor`),
roughly 88%, so it is the widest-reach knob on the entry path.

### The problem it addresses: comparability, NOT strategy

Second of two remediations for live and demo not behaving comparably:

1. `SCAN_MIN_VOL_USDT` was inert under percentile mode — the fixed floor and
   the p85 floor were both applying, so the instances screened DIFFERENT
   UNIVERSES.
2. `AUTO_CALLBACK_ATR_MULT` raised on live to 1.25 — a residual divergence in
   how often each instance entered at all.

**The metric is ENTRY RATE, not entry quality.** Until two bots see comparable
opportunity sets, comparing their trade quality measures what they saw, not
how well they chose.

### The evidence

Both at 0.75:

    demo 0.75   n=512  211.7h  2.42 trades/h  ATR% 0.825  callback% 0.700
    live 0.75   n=214  129.3h  1.65 trades/h  ATR% 0.794  callback% 0.680

ATR and callback near-identical, so the divergence was NOT volatility. Live
still entered at 0.68 of demo's rate. After raising live to 1.25:

    demo 0.75   n=24  11.7h  2.05 trades/h
    live 1.25   n=21  12.0h  1.75 trades/h      0.68 -> 0.85 of demo

### Do NOT judge this on entry quality

Live's entry-quality metrics look worse over the same window
(`roi_at_first_sight` median -0.87 -> -1.63; underwater on first sight
79.1% -> 100%). That is CONFOUNDED and is not a verdict: live's own median ATR
rose from 0.794 to 0.986 (+24%) over the same period, so the coins were not
comparable either. 21 trades over 12 hours against a 43-trade baseline from a
different day cannot separate the knob from the market.

The fair test is `roi_at_first_sight` split by ATR BAND, comparing the
214-trade historical live file against current live trades so volatility is
held roughly constant. Check band POPULATIONS before trusting it — at 21
trades some bands will have n<5, which reads as signal.

### Two open questions, deliberately not settled here

**The two remediations may not be separable.** If the `SCAN_MIN_VOL_USDT` fix
landed in the same window as the 1.25 change, the 0.68 -> 0.85 move has two
candidate causes and attribution to the callback is not established. Record
which deployed when before treating the entry-rate result as this knob's.

**The mechanism is not obvious.** A WIDER retracement floor should mean the
resting entry waits longer and fills LESS often; observed rate went UP. That
inversion is either a real second-order effect worth understanding, or a sign
the effect belongs to the other remediation.

### Current position

Keep live at 1.25 and demo at 0.75. Let the sample grow before re-evaluating.

### Consequence for anything ROI-denominated

`ENTRY_TARGET_LEVERAGE` is 10 live and 20 demo. Combined with the above, the
containers differ in entry callback AND leverage, so cross-container OUTCOME
comparison (final ROI, win rate, give-back) is not available at all. Compare
ENTRY metrics across containers; compare outcomes only within one.

---

## What v3.72.0 does to the ACTUAL deployed configs (v3.73.1)

Read this before deploying v3.72.x. The two containers differ in one setting
that now matters much more than it did.

    live:  ENTRY_TARGET_LEVERAGE=10   AUTO_CALLBACK_ATR_MULT=1.25
    demo:  ENTRY_TARGET_LEVERAGE=20   AUTO_CALLBACK_ATR_MULT=0.75
    both:  GUARD_ARM_ROI=5  GUARD_TRAIL_CALLBACK_ROI=3  AUTO_MIN_ATR_PCT=0.5

Since v3.72.1 `GUARD_TRAIL_CALLBACK_ATR_MULT` follows `AUTO_CALLBACK_ATR_MULT`.
**That chaining was a mistake** — see the AUTO_CALLBACK_ATR_MULT entry above:
that variable is the independent variable of a running experiment, so chaining
to it silently coupled an EXIT-side setting to it. Live now pins
`GUARD_TRAIL_CALLBACK_ATR_MULT=0` explicitly; demo still inherits (0.75 today).
Pin demo explicitly too, and revert the chained default to a literal. Resulting arm levels:

    LIVE (10x, 1.25)          DEMO (20x, 0.75)
    atr 0.5 -> arms at  +8.3%    atr 0.5 -> arms at  +9.6%
    atr 0.6 -> arms at  +9.5%    atr 0.6 -> arms at +11.0%
    atr 1.0 -> arms at +14.5%    atr 1.0 -> arms at +17.0%
    atr 2.0 -> arms at +27.0%    atr 2.0 -> arms at +32.0%

`AUTO_MIN_ATR_PCT=0.5` is the floor on tradeable coins, so **no live position
arms below +8.3% ROI** where it previously armed at +5%. That is the intended
consequence of refusing to trail inside noise, but it is a large change and
`GUARD_BREAKEVEN_AT_ROI=3` / `GUARD_BREAKEVEN_STOP_ROI=2` is what covers the
gap: once peak reaches +3% the stop moves to +2% and holds there until the
trail arms. Check that is behaving before assuming the window is protected.

**The two containers are no longer running the same experiment.** Different
leverage AND different floor multiplier. Do not pool their trades.

### The defect v3.72.0 introduced, fixed here

`GUARD_ARM_AT_ENTRY=true` in both. `_arm_at_entry` derived its activation from
`cfg.arm_roi` and never consulted the deferred level, so it asked for a trail
activating at +5% carrying a floored callback worth 7.5%+ ROI — engaging BELOW
entry, exactly what deferral exists to prevent. `_place_native_trail` duly
refused it (peak is ~0 at entry), and arm-at-entry silently became a no-op,
pushing every position onto the poll-driven path. On the live config that is
EVERY coin, because the floor always binds at `AUTO_MIN_ATR_PCT=0.5`.

Now `_arm_at_entry` derives its activation from `effective_arm_roi()`, and the
refusal check compares against the same level rather than `cfg.arm_roi`. The
trail is placed at entry as intended, dormant, with its activation at the
deferred level where it genuinely locks in profit.

`_vol_for` was also made fail-safe: it runs on adoption paths that can precede
full state init, and a missing ATR must cost a floor, not a position.

### Worth watching after deploy

- `ARMED AT ENTRY` lines should show activations near the table above, NOT
  +5%. If they say +5%, the fix did not take.
- `arming a X% trail ... Keeping the fixed stop instead` should be RARE. A
  burst of it means the floor is binding harder than the position can earn.
- Live's 1.25 multiplier is untested for the guardian's purpose. It was chosen
  for the ENTRY path, where it decides when to enter; the guardian uses it to
  decide how much room a running position gets. Those are not obviously the
  same problem, and if trails arm too late on live, this is the first
  assumption to question — not GUARD_ARM_ROI.

---

## Entry TIMING: three triggers, all recorded, none gating (v3.73.0)

`bot/crt.py` already held the measurement that frames this:

    79.1% of entries are ALREADY LOSING the first time the guardian sees them
    entered green -> n=9,  mean net ROI +3.60, win 55.6%
    entered red   -> n=34, mean net ROI -3.37, win 26.5%

and NOTHING in ~40 entry-context fields separated the two. So this is a
TRIGGER problem, not a screening one. The current trigger is distance — enter
once price retraces by the callback — which says nothing about whether the
move is finished.

### The live CRT numbers, and why they do not mean what they look like

    crt_agrees=False  n=6  median=-5.36  mean=-7.04  win=17%
    crt_agrees=None   n=1  median=+1.93  mean=+1.93  win=100%

**There is no True row.** CRT has never agreed with a live entry. As a gate it
would have blocked all six losers AND taken zero trades, and six negatives
with no positive class cannot distinguish an excellent filter from one that
never fires. Gating on this now would be fitting to an empty cell.

The positive class can only come from REFUSED candidates, which is exactly
what the shadow log holds — hence recording rather than gating.

### What is recorded

Every shadow row now carries `triggers`, computed in CODE from the same row
jev is shown, never asked of the model:

    crt_agrees, crt_swept, crt_side, crt_penetration_pct, crt_close_pos

None ("no opinion") stays distinct from False ("disagreed") everywhere,
including in `summarize`'s `by_crt_agrees`. Collapsing them would score a
trigger that abstained as one that was wrong — and on live data that
distinction IS the dataset.

The second trigger needs no new field: jev's `looks_exhausted` is already a
raw component on every row, and asks the same question from indicator state.
CRT and jev can now disagree on identical inputs and be scored against
identical outcomes.

### The baseline that keeps both honest

`HORIZONS_MIN` gains 3 and 6 minutes — one and two candles on the 3m
timeframe. The `wait_*` columns say the better price arrives one candle later
81% of the time (median +0.26%), so a plain delay is the baseline any
structural trigger must beat before its structure has earned anything.
`summarize()` reports it as `wait_baseline`, computed on the SAME observations
so the comparison is like-for-like rather than against a remembered statistic.

The CLI now warns on stderr when no observation has CRT agreeing, because a
trigger that never fires reads as a perfect filter in any table.

### How to read it, when there is enough

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/resolve_shadow_outcomes.py --summary

Compare `by_crt_agrees["True"]` against `wait_baseline.one_candle`. If CRT
cannot beat the plain wait, its structure is adding nothing. Same test for
splitting on `looks_exhausted`. Under ~30 observations none of it means
anything, and the CLI says so.

### Still open

- The candidate streamer is OFF (`ws=off`, REST every 3s, `STREAM DEGRADED:
  src=none`), so entries fall back to a scan snapshot up to one scanner
  interval old. That is an EXECUTION problem, not a timing one — CRT keys off
  candle closes and a faster tick feed does not help it. Worth fixing on its
  own merits; do not conflate the two.
- CRT itself is unvalidated. bot/crt.py is explicit that the source material
  was a walk-through of selected examples with no test in it.

---

## The armed trail was sized without reference to the market (v3.72.0)

STG, demo, 2026-09-20. Peak +23.2% ROI at 14:55:56. Closed -8.08% at
14:55:59 — three seconds later, realised -$10.23.

    exit_reason         "trail"        <- the trail fired; nothing failed to place
    observed_roi        18.61          observation_lag_s 2.25
    peak_roi            23.18
    final_roi           -8.08
    stop_working_type   MARK_PRICE

### What it was NOT

Two false trails, both worth recording so they are not re-run.

**Not a placement failure.** The order history shows the entry and exit as
plain `market` orders (288433048 / 288433296). That looks like a manual close,
but Binance ALGO orders (trailing stops, stop-markets) carry their own id
space and spawn a REGULAR order when they trigger — the entry proves it, logged
as algo `1000000211933431` and filled as order `288433048`. A triggered trail
always looks like a bare market order in that view.

**Not mark-vs-last, on the evidence available.** That was the first
hypothesis, and 10 closed trades across both containers do not support it:

    STG       trail      obs  18.61  final  -8.08  gap +26.69   <- outlier
    SAGA      trail      obs  18.84  final  25.84  gap  -7.00   <- divergence PAID
    SAGA      trail      obs  18.76  final  18.83  gap  -0.07
    PONS      trail      obs   2.10  final   1.93  gap  +0.17
    (+6 fail_fast, |gap| <= 1.84)

Divergence cuts both ways by side and sign — there is no systematic bias to
fix by switching working type, and one outlier cannot carry that case. Demo
also diverges far more than live (median |div| 0.151% vs 0.068%, p90 0.608%
vs 0.164% over 80 samples), so demo OVERSTATES this problem relative to live.
The tail is real but rare (AKE 2.377%, SAGA 1.164% — 47 and 23 ROI points at
20x) and is NOT addressed here.

### What it was

`trail_callback_price_pct` was `trail_callback_roi / leverage`, clamped to
Binance's 0.1%-5%. That clamp describes what the EXCHANGE accepts, not what
the market does. It never consulted ATR, recent range, or anything else about
the coin.

    STG at 19.8x:  callback 0.15%  =  0.25x its ATR (0.604%)
                                   =  0.16x its recent range (0.916%)

Meanwhile the ENTRY path (`auto_trader._callback_for`) had independently
chosen **0.45%** for the same coin seconds earlier, floored off ATR — the
`callback_source: "atr_floor"` in STG's own entry context. **Two systems sized
a callback for one position and disagreed threefold, and the guardian's
overrode the entry's.**

A trail set at a quarter of ATR fires on the instrument's ordinary breathing.
No divergence event is required.

**And the coupling ran the wrong way.** Holding give-back constant in ROI
terms makes the PRICE trail TIGHTER as leverage rises:

    5x -> 0.60%   10x -> 0.30%   20x -> 0.15%      (vs STG ATR 0.604%)

So the higher the leverage, the more certainly the trail sits inside noise.
That is why this surfaced on demo's Binance-default 20x. The live plan of 10x
halves it and does not fix it: 0.30% is still inside STG's ATR.

**This was not the first instance.** The comment at futures_guardian.py:3450
records ONE at 17:00 UTC — peak +31.8%, armed trail with a 3% ROI callback,
closed -15.18%. Same signature, diagnosed from order history days later, which
is why that exit-diagnosis line exists at all.

### The fix

`GUARD_TRAIL_CALLBACK_ATR_MULT` (default ON) floors the armed callback at
0.75x the LARGER of ATR and recent true range — the same multiplier and the
same "larger of" rule the entry path has always used, so the two systems
finally share one yardstick.

**Its default is AUTO_CALLBACK_ATR_MULT, not a literal 0.75.** The first cut
hardcoded 0.75, which matched only the OTHER setting's DEFAULT: change
AUTO_CALLBACK_ATR_MULT in compose and the guardian would have silently kept
0.75, reintroducing the very disagreement this floor exists to close. An
explicit GUARD_TRAIL_CALLBACK_ATR_MULT still wins, including 0 to disable. Volatility comes from the entry context where it
exists, falling back to live ATR for adopted positions. No volatility known
means NO floor: a floor guessed from nothing is a number with no meaning.

**The floor collides with the arm invariant, and that collision is the real
design decision here.** A noise-width callback costs more ROI at high leverage
than the +5% arm level:

    20x, ATR 0.604%  ->  callback 0.45%  ->  9.0% ROI give-back  vs  +5% arm

The old code refused to arm at all in that case and kept the fixed stop. That
is the wrong resolution — it strips the trail from exactly the volatile
positions that most need one, and at 20x it would have applied to nearly every
coin. Instead the position now arms **LATER**: `effective_arm_roi()` waits
until the peak can support a noise-width callback and still leave
`GUARD_MIN_TRAIL_LOCK_ROI` (2%) of profit.

STG replayed through it:

    callback      0.15%          ->  0.69%   (1.14x ATR, 0.75x recent range)
    give-back     3.0% ROI       ->  13.7% ROI
    arms at       +5% ROI        ->  +15.7% ROI
    armed at its +23.18% peak?   ->  YES
    trail would have sat at      ->  +9.5% ROI   (actual: -8.08%)

### Behaviour changes to expect

- Armed callbacks are WIDER on volatile coins, so the trail gives back more
  ROI than `GUARD_TRAIL_CALLBACK_ROI` names. That target is now a floor-limited
  preference, not a promise. `trail_locks_in` and `callback_roi_at` take the
  same volatility so the logs report what was actually sent — reporting the
  unfloored figure would be worse than the original bug: a wrong number that
  looks checked.
- Volatile positions arm LATER. Below the deferred level they keep the fixed
  stop and profit floor, which is the same protection they had before arming.
- The give-back is now measured against the position's ACTUAL peak, not
  `cfg.arm_roi`. A position that ran to +20% with a 10% give-back was
  previously refused a trail that would have locked in +10%; it now gets it.
  `test_a_peak_well_past_the_arm_level_DOES_get_its_trail` pins this, and the
  old test was rewritten to pin the refusal at a peak sitting ON the arm level,
  where it still holds.
- The floor is rounded UP to Binance's 2dp callback precision. Plain `round()`
  took 0.453 to 0.45 and handed back a callback fractionally INSIDE the floor.

Set `GUARD_TRAIL_CALLBACK_ATR_MULT=0` to restore the old sizing.

### Test-suite note

`_guardian()` in test_futures_guardian.py now disables the floor by default,
the same way it already disables `adaptive_trail_enabled` and `arm_at_entry` —
those are the fixed-stop path's tests, and the fake exchange's synthetic
candles produce a ~10% ATR no real coin carries. Floor behaviour has its own
tests in test_futures_guard.py. Pass `vol_floor=True` to exercise it there.

### Still open

- The divergence tail (AKE 2.377%) is unaddressed and unquantified. It needs
  many more trail exits before anything is done about `workingType`.
- `roi_at_60s`/`180s`/`300s` were null on STG because it closed in 21s. Fine
  here, but it means short-lived trades contribute nothing to that series.
- LOW ROI read 0% on the dashboard while the trade finished at -8.08%: the
  guardian never observed the drawdown it took, so the low-water
  instrumentation has a hole independent of everything above.

---

## The shadow log could not answer its own question (v3.71.0)

First real sample, 351 judged candidates:

    bot_decision   SKIP 350, ENTER 1
    jev_verdict    SKIP 351
    agree          350/351 = 99.7%
    always-SKIP baseline    = 99.7%

Agreement equal to the baseline to the decimal is zero information. Both
columns were constant, for different reasons — jev never reached ENTER
(max `composed_score` 0.51 against `_ENTER_THRESHOLD` 0.55), and the bot
entered once. The one disagreement cell held a single row.

**The threshold is NOT the thing to change.** Lowering it until jev starts
saying ENTER would tune a column into varying so the statistic looks alive.
The design was the problem: verdict-vs-verdict needs both sides to vary, and
350 of 351 rows were refusals carrying no outcome at all, so nothing could be
learned from the bulk of the data even in principle.

Note the deployment was minutes old with history cleared, and the bot was
correctly refusing per its parameters. The distributions below are a snapshot,
not a market finding — do not read the medians as characterising anything.

### Rows were not independent — 1021 of them, 24 symbols

    rows 1021  distinct 24
    CHILLGUY 106 | B2 75 | F 72 | AR 70 | C 63

The scanner re-offers the same symbols every cycle. The rate cap never fired
(volume was never the problem) so nothing flagged it. Any statistic over those
rows weights a handful of coins by how often the scanner happened to loop, and
joining them to outcomes would manufacture confidence: 106 rows about one coin
in one state share essentially one outcome.

`SHADOW_DEDUP_WINDOW_SEC` (default 900) now gates on whether the candidate is
a materially different QUESTION, keyed on `(symbol, side)`. Set 0 for the old
sampling — it changes which candidates are judged, not merely how many, so a
run compared against older rows needs it off.

**Absolute buckets were tried first and flap.** A value drifting around a
bucket edge re-keys on every crossing while nothing has changed. Measuring
against the last ASKED value instead means drift must accumulate.
Simulated over 24 symbols drifting across 180 scan cycles:

    no dedup          4320 candidates -> 4320 questions
    absolute buckets                  ->  675  (15.6%)  28.1 per symbol / 30 min
    hysteresis                        ->  136  ( 3.1%)   5.7 per symbol / 30 min

Thresholds live in `_MATERIAL_DELTA`; flags and categoricals are exact-match
since they have no noise floor. Dedup runs BEFORE the rate cap so an unchanged
candidate cannot consume budget a new one could have used.

A test also caught the dedup map being unbounded: age pruning alone cannot
bound it, because a wide scan can hold more symbols than the window expires.
Hard ceiling at 4096, oldest evicted first.

### Forward returns — what makes a refusal measurable

`bot/shadow_outcomes.py` labels each decision with what the market then did at
+15/+30/+60 minutes. The question stops being "did jev agree with the bot" and
becomes "when they refused, was refusing correct" — answerable, and what
JEV-BRIEF.md's hypothesis actually needs. It also gives the continuous
components (`conviction`, `structure_intact`) something real to correlate
against, which makes the composed verdict unnecessary rather than broken.

Entirely off the trade path. The scan row carries no price and adding one
means editing `scanner.py`, so the resolver fetches historical 1m candles and
derives both baseline and horizons itself. It builds its own read-only ccxt
client with NO API KEYS — it cannot place an order even by accident.

Outcomes go to a SIDECAR, `logs/shadow_outcomes.jsonl`, never back into the
decision log: that file is append-only and written live, and rewriting rows
in place to add a column invites a torn write for no benefit. Join on
`(symbol, ts)`.

Run it:

    docker exec $(docker ps -q -f name=scalper-pullback-1) \
        python tools/resolve_shadow_outcomes.py

Idempotent, and decisions too recent for a 60-minute outcome are left for a
later run rather than written truncated. Worth a daily timer.

**Read `observations`, never `decision_rows_covered`.** The first is the real
sample size after collapsing; the second is how many log lines fed in. On the
live sample those differ by ~40x.

`favoured_side_pct` is signed by the side, so + always means the direction was
right. Without it a winning short reads negative and every later average
silently inverts — there is a test pinning exactly that.

A test also caught the resolver carrying a stale price forward: when a feed
stopped early, 30m and 60m both returned the last known close and the gap
looked like a flat market instead of missing data. `MAX_STALENESS_S` bounds
it; a missing minute now reads as missing.

### What to do with it

Let it run a day — 30-50 entries plus refusals across several sessions — then
read the summary. Until there are ~30 observations the medians are a shape
check, not a result, and `_WEIGHTS` should not be re-fitted on them. The CLI
says so on stderr rather than trusting anyone to remember.

Two inputs still look suspect, both unresolved and both needing more than one
session to judge:

- `regime_aligned` median 0.11 across 351 rows, weight 1.0 — the heaviest term
  answering "no" almost always. Could be a short window of genuinely unaligned
  market, or a question mis-specified for short-side candidates. Read one
  row's `inputs_seen.regime` against the wording in `_QUESTION_SPECS`.
- `looks_exhausted` median 0.47, straddling the midpoint, so its derived
  confidence is ~0 and it floors every row's min. It is also the question
  closest to the entry thesis, and the one most needing the order-book data
  `inputs_seen.order_book == "UNAVAILABLE"` says it never gets.

---

## `confidence` was a dead field on every row ever written (v3.70.0)

The first successful run returned `confidence: 0.0` on all three rows. Not the
model hedging — structural. From the SDK schema:

    NoulAnswer     type, noul                      <- NO confidence field
    ScoreAnswer    type, score, confidence
    ChoiceAnswer   type, choice, confidence, probabilities

Three of the four questions are nouls. `_parse` read a `confidence` attribute
off each component with a silent `or 0.0` fallback, `min()` took that zero, and
so `confidence` was **0.0 on every row written since v3.58.0** while reading
like a measurement. The weakest-component rule — chosen deliberately over
averaging, because averaging hides the case worth abstaining on — resolved to
a constant regardless of what the model said.

### The derivation

A noul IS its own confidence. The API's schema: near 1 favours yes, near 0
favours no, near 0.5 means uncertain. So distance off the midpoint:

    _noul_confidence(noul) = 2 * abs(noul - 0.5)

    0.50 -> 0.00    0.75 -> 0.50    0.95 -> 0.90    0.05 -> 0.90

Symmetric on purpose: a confident NO is as usable as a confident YES, and only
the middle is uninformative. This puts nouls on the same 0–1 scale as the
score's real `confidence`, so min-of-components works as designed.

**Why not just drop nouls from the min** (the alternative considered): it
would measure `conviction` alone — a quarter of the judgement — and let
through exactly the row this is for. Worked example from the new code:

    AR   SKIP  score=0.543  confidence=0.04  components=[0.7, 0.04, 0.1, 0.2]

A SKIP composed right by the threshold where the model could not tell on all
three nouls. Old code recorded 0.0 and it looked like every other row.
Conviction-only would have recorded 0.7 and looked confident.

`_component_confidence` now RAISES when an answer carries neither shape,
instead of flooring to zero. A component whose confidence cannot be
established is a parse failure and belongs in the UNKNOWN path.

### The root cause was the test fake, not the arithmetic

    class _Ans:
        def __init__(self, **kw):
            self.__dict__.update(kw)

It accepted any keyword, so every test handed nouls a `confidence` the real
`NoulAnswer` has never had. `test_confidence_is_the_weakest_component` was
asserting on a field the test itself invented, and passed for twelve
versions. Replaced with `_Noul` and `_Score`, both `__slots__`-bound to the
SDK's actual shape, plus a test that reads `NoulAnswer.model_fields` directly
so a schema change on their end fails here rather than in production.

The general lesson, since this is the second instance in two versions: a fake
permissive enough to accept anything tests nothing about shape. The model-name
bug (v3.69.0) was the same failure — an injected client meant the real
constructor argument was never exercised.

### `model_release` removed — added in v3.69.0, obsolete within the hour

v3.69.0 added it because every name `GET /v1/models` offers is a floating
alias, so `model` looked unable to distinguish model eras. The first live
response settled it: the API RESOLVES the alias server-side and returns the
concrete version.

    model = 'jev-1.13.0'        <- what rows actually record
    alias sent = 'jev-latest'

So each row already names exactly which model judged it, and the release-date
proxy plus its per-process request were redundant. Both gone.

### Interpreting rows across the fix

- Rows before v3.69.0: `jev_verdict` UNKNOWN, never reached the API.
- Rows between v3.69.0 and v3.70.0: real judgements, but `confidence` 0.0 —
  ignore that column, the raw components are intact and re-fittable.
- Rows from v3.70.0: `confidence` is live. If it is ever constant across a
  run of differing judgements again, it is broken again;
  `test_confidence_is_not_constant_across_differing_judgements` pins it.

---

## SHADOW_MODEL was never a valid model name (v3.69.0)

`SHADOW_MODEL=jev` returned 400 "Unknown model: jev" on **every** shadow call
since the feature merged in v3.58.0. `jev` is the model FAMILY; every id the
API accepts carries a suffix. `GET /v1/models` offers exactly two, and both
are FLOATING ALIASES — there is nothing to pin:

    jev-latest    2026-09-10T18:38:01.391457+00:00   current general model
    jev-preview   2026-09-10T18:39:06.057655+00:00   "better in most ways"

Now `jev-latest`. NOT preview: for a measurement rig stability beats quality,
and a preview channel can move or vanish mid-run. Evaluating preview is a
deliberate A/B with the channel recorded per row, never a default.

**Why every test stayed green.** All 38 injected a fake client, and the real
one is constructed with a name the API validates SERVER-side, per call —
`TypeSafeClient(model=...)` accepts anything. The model string had therefore
never been exercised once, in any run, before production. Four tests now pin
it (`test_the_default_model_is_not_the_bare_family_name` and neighbours).

**Nothing traded on this.** The failure is on the shadow path, caught in
`_run`, logged UNKNOWN. It never gated, sized or delayed anything. The cost
was a poisoned dataset, not money.

### Three defects, not one

**1. The breaker never fired.** `_client_broken` was set only when the client
failed to CONSTRUCT, which a bad model name does not. So every candidate
spawned a thread and fired a request guaranteed to 400 — up to
`SHADOW_MAX_PER_MINUTE` (30) a minute, indefinitely, at WARNING level where it
just accumulated. `_run` now treats a message containing "Unknown model" as
fatal and latches, since retrying cannot fix a config error. Transient
failures (502 etc.) deliberately still do NOT latch; both directions tested.

**2. The breaker gated nothing even once set.** Found by the new test, not by
inspection. `_get_client` opened with

    if self._client is not None or self._client_broken:
        return self._client

so once a client was cached the flag was read but the client handed out
anyway. The breaker check is now first and returns None. This bug predates
v3.69.0 and would have defeated ANY latching added downstream.

**3. `model` cannot distinguish model eras.** Rows record
`getattr(result, "model", self.model)`. Since every available name is a
floating alias, that string is identical on every row forever — including
across a silent swap of what the alias points at. Re-fitting `_WEIGHTS`
offline would then average rows judged by different models with nothing to
separate them, which is the exact failure the raw-component recording exists
to prevent.

`model_release` is now written on every row: the alias's `release_date`,
captured once per client construction (not per judgement — one extra request
per process start). It belongs to whatever the alias resolves to today, so it
MOVES when the alias moves. Group by it offline: constant across the file
means the comparison is clean; a change means you have found the boundary.

Never fatal — an unreadable stamp costs provenance, not judgements, and rows
then carry `model_release: null`.

### Two things to check on first real run

- **Log `result.model` from a live response.** If the API resolves the alias
  to a concrete version rather than echoing `jev-latest` back, that is
  strictly better than the release_date proxy and `_capture_model_release`
  can be dropped. Unverified — no live call has been made.
- **`release_date` arrives as a full timestamp** despite the API's own schema
  documenting `YYYY-MM-DD`, and is typed as a bare `str`, so pydantic passes
  it through unvalidated. Stored verbatim. Treat it as an opaque key; do not
  parse it as a date without checking the format still holds.

### Still open

Existing `jev_verdict: "UNKNOWN"` rows written between v3.58.0 and this fix
are all unreachable-model failures, not abstentions, and are indistinguishable
from real ones in the file. `reasons[0]` carries the exception text, so they
can be filtered on that — but consider a distinct `UNREACHABLE` verdict so
the two stop sharing a value. Not done here: it changes the on-disk schema
and `agrees_with_bot()`, which is a bigger change than a config fix warrants.

---

## The finding

On five shorts across 2026-09-17/18 the guardian's price came in **below the
real mark every single time**. For a short that inflates ROI. The damage
scaled with the size of the gap, which is why it looked like four unrelated
bugs.

Binance's activation prices are the latest mark at placement, so the order
history gives the real number independently of the logs:

| trade | guardian price | real mark | error | in ROI |
|---|---|---|---|---|
| COTI 18:04 | 0.020528 | ≥0.020551 | +0.11% | +2.2 pts @20x |
| ONE 19:56 | 0.0015552 | ≥0.0015585 | +0.21% | +4.2 pts @20x |
| OP 03:16 | 0.110685 | 0.1117961 | +0.99% | +9.9 pts @10x |
| 牛来 04:23 | 0.10926 | 0.1211 | +9.78% | +97.8 pts @10x |

Consequences by magnitude:

- **0.11%** — floor refused with -2021, retried lower, survives. COTI closed
  +1.46%.
- **0.21%** — floor refused; the trail is accepted but its activation sits
  below mark, so it rests **dormant** until price falls to it. ONE 19:56 shows
  this; it eventually Finished.
- **~1% and ~10%** — the activation is so far below mark that Binance refuses
  the trail outright, and the reported ROI is fiction. OP was logged at +9.8%
  while actually -0.14%. 牛来 was logged at **+96.5% while 1.41% DOWN**.

Neither OP nor 牛来 was ever a position in deep profit that gave it back. OP
closed -2.14% (fees plus 0.26% slippage on the floor fill) having never been
more than 0.3% ahead. 牛来 closed **+0.71%**.

`peak_roi` is monotonic, so one bad reading pins it for the life of the
position and every later decision is taken against it.

---

## What changed

1. **`_price_from_ticker` returns MARK**, via `_mark_and_last()` and
   `_premium_index_mark()`. ccxt's binanceusdm ticker comes from
   `/fapi/v1/ticker/24hr`, which carries no `markPrice`, so the premium index
   is the normal path rather than a fallback. Cached 1s
   (`MARK_CACHE_TTL_S`) — briefly, because a stale mark is the thing this
   exists to avoid.

   Falls back to `last` with a warning when no mark exists anywhere. Acting on
   last beats going blind, but every level that cycle is then computed against
   a reference the exchange does not trigger on.

2. **Every price now names its source.** `self._last_price_source` is set on
   each read. `resolve_price()` reaches for `previousClose` (24 hours old) and
   a completed candle close before giving up, and anything that is not a live
   price is logged at WARNING: *"price X came from Y, which is NOT a live
   price"*.

3. **`PRICE-DIVERGENCE`** logged when mark and last differ by ≥0.05%,
   throttled to 30s per symbol.

4. **`PEAK` logged whenever `peak_roi` advances**, with the price, its source,
   the entry and the leverage. This is what turns "+96.5% appeared from
   somewhere" into a traceable line.

5. **An order the exchange refused is no longer recorded as protection.**
   `_accepted_id()` checks the status on the response — Binance answers 200
   with the order body either way, so `create_order` returning normally is not
   acceptance. Applied at all four placement sites: the fixed stop, **each leg
   of a split stop**, the rescue trail and the armed trail.

   OP `03:16:27` and 牛来 `04:23:07` were both logged as *"ARMED native
   trailing stop ... locks in ~+2% ROI"* with an id. Binance shows both
   REJECTED. The guardian then cancelled the adaptive trail as "superseded" in
   favour of an order that did not exist, leaving the position bare until a
   floor landed seconds later. Returning None makes callers treat it as a
   failed placement, which is what they already do for an exception, so the
   supersede cannot fire.

The split-stop legs were missed on the first pass and caught by a test that
counted the call sites.

6. **The Peer button is hidden unless the peer is live.** `/api/status` now
   carries `peer_eval_enabled`, derived from whether the PeerEval object
   actually sends or receives rather than from the mode string — `sends` also
   requires `PEER_EVAL_URL`. The button was static markup with no gating, so
   with `PEER_EVAL_MODE=off` it stayed in the toolbar and only ever answered
   "Peer evaluation is not active". It also now carries the same
   `padding:3px 6px` as the window and export selects either side of it, which
   it was missing. A backend without the field is treated as off.

---

## P&L source ordering (v3.43.0)

`a9b1d41` "Fix reconciliation" put the income ledger above everything because
a price ESTIMATE invented +4.32 USDT on a trade opened and closed at the same
price. **That reasoning stands**, and is why the estimate is still last.

What it also did was put the ledger above the FILLS, and those are not an
estimate. WLD 2026-09-18 10:16:04, three independent records:

    fills sum ............ +0.2684
    Binance order detail . Total PNL 0.26840000 (order 23149661775)
    wallet moved ......... 89.97 -> 90.23 = +0.26
    income ledger ........ +0.0934   <- taken, and wrong

Roughly a third of the true figure: a partially-populated income query on a
multi-fill close, read seconds after the position closed. It cost that trade
0.175 USDT of recorded profit and reported +3.57% ROI instead of about +10.2%.

The order is now:

1. **fills**, when they cover the position (closing-side quantity / position
   quantity >= 0.99)
2. **ledger**, when fills are missing or short
3. **computed**, last — the estimate, which is the only one that can invent
   money

The original bug cannot return through fills: a trade opened and closed at the
same price sums to zero in the fill records, correctly.

**Fees still come from the ledger.** `_income_pnl` converts BNB commission via
`_fee_asset_rate`; summing BNB as USDT reported 0.0000 on every live trade, so
live was silently gross while demo was net. Nothing routes around that.

The disagreement is still logged in both directions. That line is what
surfaced this.

**Success criterion, already built:** `analysis.py:438` computes
`wallet_gap = moved - net`, shown in the UI header as the gap between
ACCOUNT RETURN and the bot's own figure. It read **+$3.15 across 134 trades**,
the bot under-reporting. If this fix is right, that converges toward zero. If
it does not, something else is also mis-recording.

## Trail activation and the audit (v3.44.0)

Both were detected in v3.42.0 and only now corrected. Until this version the
guardian could *see* a dormant trail and did nothing about it.

**Trails now carry an explicit activationPrice.** Sending none does not
activate at the mark — Binance derives its own. WLD 2026-09-18 07:13:47 went
on 0.043% short of reachable:

    ARMED trail activation=0.4300807 mark=0.43026546 (-0.043%)
        -> DORMANT until price reaches it
    (one second later) adaptive trail superseded — cancelled

A dormant trail with the adaptive one already cancelled leaves only the profit
floor, and below GUARD_BREAKEVEN_AT_ROI there is no floor either. WLD survived
because price kept moving in its favour, which is luck.

`_activation_now` puts the activation on the already-satisfied side of the
mark: above for a BUY (closing a short), below for a SELL.
`GUARD_TRAIL_ACTIVATION_EPS_PCT` defaults to 0.3%, comfortably past the 0.043%
that caught WLD and the 0.010% that nearly caught ENA.
`GUARD_TRAIL_ACTIVATE_NOW=false` is the kill switch.

**If the exchange refuses the explicit activation, the placement retries
without one** — which is exactly the old behaviour, so this cannot leave a
position with less protection than before, only with a trail live sooner.

**The audit now reads the algo book.** `_audit_protection` called
`fetch_open_orders`, the unified book, which structurally cannot hold
conditional or trailing stops. `_open_orders_multi` already documented this;
the audit did not. WLD 07:15:20-24, one pass: unified 0, raw fapi 0, algo 2 —
while PROTECTION-BLIND reported "nothing at all". Costs no protection, only
sight: a stop that silently vanishes now gets noticed.

## Entry-factor grouping (v3.45.0)

Analysis only. No behaviour change, no risk to the bot.

G/USDT 2026-09-18 08:02 was shorted after **+68.78% in 24h** at `atr_pct`
3.203, with a 30% ROI stop — 1.5% of price at 20x, **less than half one
average candle**. It closed at -30.3%, never having been positive.

Neither factor could be grouped:

- `ATR_BUCKETS` topped out at an open-ended `1.5%+`, so a 1.6% coin and a 3.2%
  coin shared a bucket. Now `1.5-2.5%`, `2.5-4%`, `4%+`.
- `change_24h_pct` was captured per trade (`analysis.py:568`) and never
  bucketed. New `by_change_24h`: `<0%`, `0-10%`, `10-25%`, `25-50%`,
  `50-100%`, `100%+`. Rendered beside the ATR table.

**The question these exist to answer.** ATR is gated BELOW by
`AUTO_MIN_ATR_PCT` (`auto_trader.py:349`, and `min_atr` is the top refusal in
nearly every cycle) and **not above** on the auto-trade path — `SCAN_MAX_ATR_PCT`
exists only at the scanner. Meanwhile `callback_atr_mult` raises the callback
to 0.75x volatility, so a higher-ATR coin demands a deeper retrace to enter
(G: 0.25% -> 2.4%).

**Do not assume the answer.** `auto_trader.py:208-213` records that on 64
velocity-floored trades the nine LONGS returned -$17.64 each while the 55
SHORTS returned +$1.65, concluding it is ATR-based widening of a LONG's
callback that hurts. G was a short. So "high ATR plus a widened callback is
bad" is already contradicted for shorts by a 64-trade sample.

**Caveat on the history.** v3.44.0 changed protection behaviour, so post-deploy
trades are not comparable to earlier ones on anything peak-related. Pre-deploy
trades carry the phantom-peak contamination. Entry context (ATR, 24h change)
is unaffected by both, so the entry question can be asked of the whole record —
but exit-side statistics cannot.

## Short entry confirmation (v3.46.0)

**The intent was to confirm a downturn at entry. Nothing tested for one.**
`scanner.py:791` states it outright: convergence is "INFORMATIONAL
(fade-early mode). RSI leads the screen." A short qualified on RSI plus
`gap > -ema_tolerance_pct` — which a strong uptrend passes by construction,
since it excludes coins whose EMAs have already crossed DOWN, not coins still
trending up.

The confirmation was already computed and thrown away. `turned(df, cfg,
"short")` asks whether the HIGH is behind us and is carried on every short
candidate as `turn`. G/USDT 2026-09-18 07:59 recorded **turned_up=False,
bars_since_low=0** — the extreme was the CURRENT bar, still making new highs —
and lost 30% ROI. The long side has had `long_require_turn` (default True) all
along; the short side had no equivalent.

Two new screens, mirroring the long ones:

- `AUTO_SHORT_REQUIRE_TURN` — the extreme must be behind us. **Default FALSE.**
  Not timidity: switching it on makes 26 existing short tests refuse entries
  they used to take, which is the measure of how much it bites. Set it true in
  compose and compare one run against one run.
- `AUTO_SHORT_REQUIRE_CONVERGENCE` — EMA9 must be falling back toward EMA21.
  Default FALSE, and note G **passed** this one (gap_narrowing was True), so
  enabling both at once would prove nothing about which did the work.

Watch the `no_turn` refusal counter for shorts once enabled.

## Audit false alarm (v3.46.0) — my regression

v3.44.0 made the audit read the algo book. Binance's algo rows do not carry
`reduceOnly`, so they failed the protective test and G 2026-09-18 08:38:39
reported "2 order(s) but NONE is protective" with both tracked ids MISSING —
while both were resting fine. Worse than the PROTECTION-BLIND it replaced,
because it asserted a false negative.

Existence and classification are now separate questions. `all_ids` covers every
book for the MISSING check; the protective test stays STRICT on purpose,
because entries here are TRAILING_STOP orders too and widening it would let the
sweep cancel one.

## Mark lag on a spike — open

G 08:38:33: `mark=0.007801 last=0.007964 (-2.089%)`. At 10x that is 20.9 ROI
points, and GIVE-BACK fired at 18.1 — "MORE THAN THE TRAIL CAN EXPLAIN". The
peak of +12.3% was measured on a lagging mark while the tradeable price was 2%
higher. Stops trigger on mark but FILL at last. This is the mark/CONTRACT_PRICE
question with a concrete case attached, and it is still open.

## Arm the trail at entry (v3.47.0) — OFF BY DEFAULT

**The problem.** The guardian polls every 2.5s. ROI can reach +10% and fall
back to +2% between two polls, and the trail never arms — because arming was
an event the guardian had to WITNESS. Binance sees every tick.

**The fix.** Place the armed trail AT ENTRY, dormant, with its
`activationPrice` at the arm-ROI price. `price_for_roi(pos, arm_roi)` lands
below entry for a short and above for a long, which is exactly the side each
trail activates from — no epsilon, and nothing to outrun. Contrast
`_activation_now`, derived from a mark read moments earlier: PENDLE
2026-09-18 12:38:49 lost that race by 0.023% while price moved 0.323%.

Strictly earlier, never later. The trail protects nothing until price reaches
the arm level, which is exactly when the old code would have placed it.

**No stacking.** One trail per role. `_arm_at_entry` writes `native_trail_id`,
and the poll arm block is already gated on `not state.native_trail_id`, so it
becomes a no-op — including its supersede of the adaptive trail.

**The adaptive trail therefore LIVES.** That is a gain: every supersede
discarded the extreme Binance had been tracking on it, and that extreme is the
peak the guardian cannot see. Both orders are reduceOnly and close the same
direction; whichever fires first closes the position and the other is
rejected harmlessly.

**The profit floor is unchanged and still poll-dependent.** It cannot be
pre-placed: for a short it is a buy-stop BELOW entry, so at entry it is on the
wrong side of the market — that is the -2021, not a bug. The +3% to +5% band
stays exactly as exposed as before.

**The old floor-fallback is now inert**, not removed. It set `state.armed`, and
with a trail already resting the arm block no-ops. Harmless; left in place as
the fallback when arm-at-entry is off.

**DEFAULT FALSE.** Enabling it makes 15 existing tests fail — all assertions
about ORDER SEQUENCE (stop first, trail on arming), not functional
regressions, since the trail now lands earlier in the lifecycle. Those 15 were
NOT rewritten, so the enabled path is unit-tested (11 new tests) but not
integration-tested. Weigh that before turning it on.

    GUARD_ARM_AT_ENTRY=true

**Dormancy is no longer an alarm** for an arm-at-entry trail — the
TRAIL-ACTIVATION line drops to INFO "dormant by design" for that case only,
so it does not cry wolf on every entry. An unintended dormancy still warns.

## ENTRY_TARGET_LEVERAGE — recommended ON

The live account is NOT running at 10x. Across 44 trades with
`ENTRY_ASSUMED_LEVERAGE=10`: **18 at 10x, 26 at 20x**, and G ran at both on
the same day. `ENTRY_ASSUMED_LEVERAGE` is only a FALLBACK for when the
exchange reports nothing; otherwise the bot reads whatever leverage Binance
has per symbol and sizes for that.

It matters through the stop, not the margin. `raw_roi = atr_stop_mult *
atr_pct * leverage`, so the stop is normally 1.5x ATR whatever the leverage —
self-correcting. Except when `ATR_STOP_MAX_ROI=30` binds, and then leverage
leaks through:

    10x: 18 trades, stop 1.48x ATR, cap hit 2 times
    20x: 26 trades, stop 1.38x ATR, cap hit 8 times

    G    20x atr 3.20%  stop 0.47 ATR   <- half an average candle
    ONE  20x atr 3.09%  stop 0.49 ATR

Set live to 10 and demo to 20 — that also makes the two comparable for the
first time. Two cautions: `futures_entry.py:309` records that this previously
raised `'EntryService' object has no attribute 'exchange'` and STOPPED ENTRIES
ENTIRELY (fixed, but watch the first cycles), and `ensure_leverage` writes to
the account, so the setting outlives the container.

## TradFi perps must be kept out of the scan (v3.48.0)

SNXX 2026-09-18 14:14:56, and again at 14:15:36:

    ENTRY REJECTED SHORT SNXX/USDT:USDT: -4411
      — Please sign TradFi-Perps agreement contract fapi.

Binance's TradFi perps (tokenised equities) share the ticker feed and pass the
volume and RSI filters like any other mover. -4411 is an ACCOUNT PERMISSION,
so retrying never helps — and a rejection opens no position, so it triggers no
cooldown, so the symbol is re-attempted every scan indefinitely. Each attempt
also CHANGES THE SYMBOL LEVERAGE before being refused.

Three defences, in the order they fire:

0. **Class filter, BEFORE anything is attempted (v3.50.0).**
   `SCAN_ALLOWED_UNDERLYING` (default `COIN`) filters the universe on
   Binance's exchangeInfo `underlyingType`. Crypto perps are COIN; TradFi
   perps are not. Applied in BOTH `_prefilter` loops, so they never reach the
   candidate list and never move the percentile volume floor.

   **FAILS OPEN**, deliberately: a symbol with no `underlyingType`, an
   unloaded market, or an exchange that raises all pass through. Filtering the
   wrong way would silently empty the universe, which is far worse than
   letting one refusable symbol through.

   **The value was not verified against Binance** — the container cannot reach
   it. So the scanner logs a one-time inventory per process:

       instrument classes in the scan universe:
         underlyingType=COIN: 700 symbol(s), e.g. INJ/USDT:USDT | ...

   Read that line on the first scan. If crypto perps show a value other than
   COIN, widen `SCAN_ALLOWED_UNDERLYING`; if TradFi perps share COIN, this
   filter cannot separate them and defence 2 is what catches them.

1. **`SCAN_EXCLUDE_SYMBOLS`** (optional now) — comma-separated BASES (`SNXX`, not
   `SNXX/USDT:USDT`). Filtered out of `_prefilter` in BOTH loops: the mover
   selection AND the percentile volume sample, so an untradeable symbol cannot
   move the volume floor either.

   Note the older `BLACKLIST` is consulted only by the SPOT engine
   (`engine.py:334`) and has never applied to futures.

2. **Automatic block of the whole INSTRUMENT CLASS (v3.49.0).** A per-symbol
   list needs maintaining and the next listing defeats it, so one rejection
   now teaches the class.

   On a permanent rejection the bot reads the refused symbol's ccxt market
   info and fingerprints it on Binance's own exchangeInfo fields —
   `underlyingType`, `underlyingSubType`, `contractType`, `marginAsset`,
   `quoteAsset`. Every candidate sharing that fingerprint is skipped WITHOUT
   being attempted. TradFi perps carry a different `underlyingType` from
   crypto perps, so blocking SNXX blocks its siblings and leaves INJ alone.

   Safety properties, each with a test:
   - an unknown or unreadable market fingerprints to None, and **None never
     matches** — otherwise one rejection would block everything
   - an exchange that raises on `market()` returns None rather than breaking
     the scan
   - nothing is blocked before a rejection actually happens

   Matched wording is narrow: `-4411`, "agreement", "not authorized", "not
   permitted", "permission". `-2021` and margin errors stay retryable.
   Process-scoped, so a restart clears it and signing the agreement takes
   effect on its own.

   `SCAN_EXCLUDE_SYMBOLS` remains as an optional pre-emptive override, not a
   requirement.

**Still open:** the leverage write happens BEFORE the order is placed, so a
symbol that is about to be refused still has its leverage changed on the
account. Reordering that touches the entry path and was not done here.

## ARM-AT-ENTRY IS NOT WORKING AS INTENDED — investigate before trusting it

G/USDT 2026-09-18 14:46:43, same order id, one millisecond apart:

    TRAIL-ACTIVATION ... activation=0.0098795  mark=0.00987957  -> DORMANT
    ARMED AT ENTRY   ... trail resting with activation at 0.009818 (+5% ROI)

We asked for **0.009818**. The exchange kept **0.0098795**, which is mark to
within 0.0007%. Our level was 0.623% away.

So the trail is live essentially AT ENTRY with a 0.3% callback, not dormant
until +5%. On a coin at `atr_pct 2.812` that is a trail an eighth of an
average candle wide. It did not fire on G, but it could.

**No rejection was logged**, so the retry path never ran: the order was
ACCEPTED and the activation SUBSTITUTED. That is the dangerous shape — nothing
raises, and the trail rests where nobody put it.

**SOLVED: the field is `activatePrice`, not `activationPrice`.**

ccxt maps `activationPrice` because that is the field for
`POST /fapi/v1/order`, then routes conditional linear-swap orders to
`POST /fapi/v1/algoOrder` (`binance.py:6888`) — a different endpoint with a
different schema. Binance IGNORES unrecognised parameters rather than
rejecting them, so every trail between v3.44 and v3.54 was accepted with its
activation silently defaulted to the current price.

Proof, COTI demo 2026-09-18:

    activatePrice=0.02181  mark=0.020771  ->  kept 0.0218100 EXACTLY (+5.002%)
    reduceOnly=True, workingType=MARK_PRICE

That is the SAME combination substituted eight times running under the other
spelling. The response has always returned `activatePrice`; that was the clue,
and it sat in the logs unexamined for hours.

**What this cost.** Four hypotheses were proposed and eliminated at length —
a direction constraint, ccxt dropping the parameter, `reduceOnly`,
`workingType`, then a minimum distance — each supported by real evidence and
each wrong. They were eliminated by the operator testing in the Binance UI and
by probe runs, not by reasoning. The answer was a spelling.

**v3.54.0 changes:**

* `_create_trail_order` sends `activatePrice`.
* **`_activation_now` is DELETED.** It computed an activation just past the
  mark for "live at once". It never worked (wrong key), and now the key is
  right it would be HARMFUL: Binance requires a BUY trail's activation at or
  below the current price, so a value just above would be REJECTED rather
  than quietly dropped. "Activate now" is expressed by OMITTING the field —
  Binance then defaults to the current price, which is the same thing.
* `GUARD_TRAIL_ACTIVATE_NOW` and `GUARD_TRAIL_ACTIVATION_EPS_PCT` removed from
  all four layers. They fed only the deleted helper.
* Only an explicit LEVEL is ever sent, which is `_arm_at_entry`'s arm-ROI
  price from `_activation_at_roi`.

**ARM-AT-ENTRY SHOULD NOW ACTUALLY WORK.** It has never been tested with a
working activation. Still `GUARD_ARM_AT_ENTRY=false` by default; enabling it
still fails 15 order-sequence tests that were never rewritten. Watch
TRAIL-RESPONSE on the first armed entry: `sent` and `kept` should now match.

## PROTECTION-OVERLAP was crying wolf (v3.51.0)

Under arm-at-entry BOTH trails rest by design: there is no arm event, so the
adaptive trail is never superseded — and not superseding it is the point,
since every supersede discarded the extreme Binance had been tracking. The
check now logs at INFO when `arm_at_entry` is on, and still WARNS when it is
off.

## The daily baseline drifted, and RE-BASE could not fix it (v3.55.0)

**Why "from $5488" kept moving.** `day_report` never read a stored balance. It
computed `base = wallet_now - net_since_local_midnight`. That is a
SUBTRACTION, so it absorbs every per-trade recording error, cumulatively. Four
consecutive demo closes on 2026-09-18:

    wallet moved   recorded net    error
      -21.5700       -23.5007     +1.9307
      +29.2600       +28.1583     +1.1017
       +5.4200        +2.2620     +3.1580
       -3.6400        -4.2795     +0.6395
                      cumulative  +6.8299

Over 47 trades the card drifted 5504 -> 5488, which is why it read like a
rolling 24h figure. The same quantity appears account-wide as `wallet_gap`
(-$306.93 in the header). Sources: fills-vs-ledger disagreements, BNB fee
conversion, funding, and trades excluded by `_verified()` that still move the
wallet.

**Why RE-BASE did nothing.** It cleared `_DAY_BASELINE` and zeroed
`day_start_balance`. Clearing a cache of a DETERMINISTIC function recomputes
the identical number, so the card never changed. There was no stored figure to
reset.

**And it was quietly dangerous.** Zeroing `day_start_balance` moved the DAILY
HALT's threshold (`auto_trader.py:768`) to the current balance — pressing
RE-BASE while down 5% erased the drawdown with no visible sign. That matters
now `AUTO_DAILY_HALT_ENABLED` is being considered.

**The fix: observe the balance, do not derive it.**

* `SafetyState` gains `day_baseline_at` and `day_baseline_source`
  (`rollover` | `restart` | `operator`), both persisted.
* `roll_day` marks a capture `rollover` only when it happens within
  `BASELINE_FRESH_S` (300s) of local midnight — i.e. the process was actually
  running. Waking up mid-day is `restart` and is NOT the day's start.
* `day_report` uses a STORED baseline when the source is `rollover` or
  `operator`. No arithmetic, so no drift. Reconstruction remains the fallback
  for a container that was down overnight, cached per day key so it at least
  stops moving between refreshes.
* RE-BASE now WRITES: it derives today's 00:00 balance once, stores it, and
  marks it `operator`. The card and the halt then share one figure.

The day boundary itself was always correct — `day_start_ts` and `_day_key`
handle UTC+3 properly. Only the balance attached to it was wrong.

**Still true:** a `restart` baseline hands the halt back its allowance, as
`futures_state.py:14` warns. Reconstruction cannot see deposits or
withdrawals.

## activatePrice CONFIRMED WORKING IN PRODUCTION (v3.55.1)

MAGMA demo 2026-09-18 20:56:24:

    TRAIL-REQUEST  params={... 'activatePrice': 0.24505}
    TRAIL-RESPONSE activatePrice sent=0.24505 kept=0.2450500
    ARMED AT ENTRY — trail resting with activation at 0.24505 (+5% ROI)

Exact. Arm-at-entry now genuinely rests dormant at the arm level, and the
RESCUE trail correctly sends NO activatePrice (`sent=None`), letting Binance
default it to the current price.

## RE-BASE crashed on first use — my bug (v3.55.1)

    day: failed: name 'DAY_TZ_OFFSET_H' is not defined

`api.py` imports `DAY_TZ_OFFSET_H` PER FUNCTION (`:240`), not at module level.
The new day branch used it without importing it. Every test read the source
rather than running it, so nothing caught a NameError.

Fixed, and `test_the_reset_endpoint_imports_every_name_it_uses` now parses the
endpoint's AST and asserts the names it uses are imported inside it. The
endpoint was also executed directly to confirm it returns
`5000.00 -> 5114.00 (stored)` rather than raising.

**Lesson for this file:** source-inspection tests cannot catch runtime errors.
Several of the tests here assert on `inspect.getsource(...)`; they are cheap
and they verify intent, but they will not tell you the code runs.

## ARM-AT-ENTRY SUPPRESSED THE FIXED STOP (v3.56.0) — regression, now fixed

`manage_position` did:

    self._arm_at_entry(pos, state)      # sets native_trail_id
    ...
    if state.native_trail_id:
        return                          # everything below is skipped

That early return was written for the OTHER order of events — fixed stop
first, trail armed later, trail supersedes stop. Arm-at-entry inverts it, so
on the first cycle the return fired before the initial fixed stop was ever
placed. Positions ran with TWO TRAILS AND NO FIXED STOP.

Visible in the live logs. 龙虾 2026-09-19 03:25:08 shows the SIZED handoff,
both trails, and NO "initial protective stop at -10.3% ROI" line — compare INJ
10:49:17 before arm-at-entry was enabled, which has it.

Not unprotected: the adaptive trail is live from adoption and caps the loss at
the ATR stop distance. But the belt-and-braces fixed stop silently stopped
being placed from the moment GUARD_ARM_AT_ENTRY was set true.

**Fix:** new `GuardState.armed_replaced_stop`, persisted, set ONLY where the
armed trail genuinely cancels a fixed stop. The early return now reads
`native_trail_id and armed_replaced_stop`, so it means what it always
intended: the trail owns protection BECAUSE it replaced the stop.

**How it was found, and how it should have been.** When arm-at-entry shipped,
15 tests failed and were dismissed as "order-sequence assertions, not
functional regressions". They were order-sequence assertions BECAUSE THE ORDER
MATTERS. Rewriting them instead of reasoning about them would have caught this
immediately.

`_guardian()` in test_futures_guardian.py now disables arm_at_entry by the
same rule it already used for the adaptive trail — those are the poll-arm
path's tests — and seven new full-cycle tests cover the enabled path:
placement order, activatePrice on the correct side for both directions,
reduceOnly on everything, no double placement across cycles, the off switch
restoring the old sequence, a failed trail not blocking the stop, and this
regression.

**GUARD_ARM_AT_ENTRY now defaults TRUE**, matching the deployment. A default
that disagrees with the running config is a second thing to reason about.

## The daily loss limit trails the day's HIGH (v3.57.0)

`AUTO_DAILY_LOSS_LIMIT_PCT` measured drawdown from the OPENING balance, so a
day that ran +3% and bled back to breakeven had "lost nothing" and kept
trading, having handed back the entire day.

It now trails `state.day_peak_balance`, seeded at the open, updated every
check, reset at rollover, persisted. On a day that only falls, peak == open
and the two rules are IDENTICAL — never looser, only tighter on days that
were ahead.

The halt message reports both: drawdown from the high AND the day's return,
the latter signed as a return so "-3.0% on the day" reads correctly (it was
printing +3.0% for a day that was down; caught by test).

## Funding was never counted (v3.57.0)

Binance breaks Realized PNL into **Closing PNL + Funding Fee + Trading Fee**.
`_income_for_position` read REALIZED_PNL and COMMISSION; `grep FUNDING` across
bot/ returned NOTHING. Scalps rarely straddle a funding stamp, but when one
landed it moved the wallet and nothing recorded it.

Now folded into the realised figure, as Binance's own card does. **Signed, not
abs()** — funding is received as often as paid.

**Correction to an earlier claim in this file:** BNB commissions are NOT
dropped. `_fee_asset_rate` converts them, which is why live records
`fee_source=converted` on all 43 trades and demo records `ledger` on all 51.
That difference is asset, not a bug.

## Account value vs sizing balance (v3.57.0)

On an account funded 4500 USDT + 500 BNB, every fee leaves the BNB balance and
never touches USDT. A USDT-only figure cannot see fees AT ALL, so the trade
record drifts from the wallet forever — a contributor to `wallet_gap`.

`resolve_account_value(bal, prices)` sums every asset's walletBalance at its
own price and returns `(value, per_asset, unpriced)`.

**`resolve_usdt_balance` is UNCHANGED and sizing still uses it.**
`ENTRY_RISK_PCT` means a fraction of the capital that can absorb a LOSS, and
the fee reserve cannot take one — sizing against it would risk money that is
not at risk. Demonstrated by test: 0.16 BNB of fees moves account value by
-100.00 and leaves the sizing balance identical.

An asset with no price is REPORTED in `unpriced`, not silently dropped, since
dropping understates the account with no signal.

**Still to verify on the live account** (cannot be checked from here): whether
`totalWalletBalance` in Single-Asset Mode is USDT-only or already a composite.
Read `assets[]` and compare. It decides whether sizing has quietly been using
a composite figure all along.

**Not yet wired to the cards.** The resolver exists and is tested; nothing
calls it. The UI still shows the USDT figure.

## Shadow decision log merged (v3.58.0)

`bot/shadow_decision.py` + `/api/shadow`, built to JEV-BRIEF.md. For every
AUTO-ENTRY candidate — entered OR refused — an advisory ENTER/SKIP verdict is
logged to `logs/shadow_decisions.jsonl` BEFORE the outcome is known, joined
later to the closed trade by `entry_context.entry_order_id`.

**Three lineages existed and this merged only one.** v3.57.0 (this line), the
jev shadow build, and the jev symbol-screen build were all forked off
v3.56.0. The shadow build carried v3.56.0's daily-limit code, so merging it
wholesale would have REVERTED the trailing-high limit. Applied the other
direction instead — shadow's additions onto v3.57.0 — and verified after:

    OK  daily limit trails the high      OK  activatePrice
    OK  peak persisted                   OK  fixed-stop regression fix
    OK  funding folded in                OK  account value resolver

**The symbol screen was NOT merged.** In 44 live trades there was not one
stablecoin or wrapped token, and AUTO_MIN_ATR_PCT=0.5 excludes pegged assets
by construction (lowest ATR traded: 0.512%). With SCAN_ALLOWED_UNDERLYING
already removing 199 of 725 symbols, it has nothing to filter.

**Verified it cannot gate a trade**, in the merged build, not the donor:

* 2 call sites, both inside try/except, both guarded by
  `getattr(self, "shadow", None)`
* no verdict is ever read back — structurally incapable of influencing a
  decision
* `_run` raising -> caller returns in 0ms, no exception propagates
* `_run` hanging 30s -> caller returns in 0ms

OFF by default (`SHADOW_ENABLED=false`), needs `TYPESAFE_API_KEY` and the SDK,
30 calls/min cap.

**Known limitation:** capped candidates are absent from the log rather than
recorded as UNKNOWN, so `n_trials` counts decisions MADE, not candidates SEEN.

## FUTURES_STATE_RESET did not clear the trade history (v3.59.0)

Closed trades moved to `logs/trades.jsonl` when the journal was introduced —
`futures_guardian.py:2585`, "the journal is the source of truth" — and
`futures_state.reset()` was never updated. It cleared
`data["closed_trades"]`, a field nothing reads any more.

So the reset HALF-FIRED, and reported success either way. Observed
2026-09-20 after `FUTURES_STATE_RESET=history`:

    512 trades, +$1605.64 P&L, -$1124.27 fees   SURVIVED (journal)
    wallet_start / ACCOUNT RETURN +0.00%        CLEARED  (state file)

`reset()` now takes `journal_path` and archives the journal for BOTH modes,
renamed rather than deleted like the state file. `main.py` passes
`cfg.trade_journal_path`, falling back to `trades.jsonl` beside the state
file — the same expression the guardian uses to attach it, so they cannot
disagree.

Unchanged and deliberate: `history` still KEEPS `day_start_balance`,
`day_peak_balance` and open positions, so a reset cannot hand the daily halt
back its allowance mid-day. Only `wallet_start` resets, because P&L
measurement restarts with it.

A missing journal, or a missing state file, is not an error — and the journal
is archived even when the state file is already gone.

**Still NOT cleared by either mode:** `logs/shadow_decisions.jsonl`. Delete it
by hand if a clean shadow log is wanted.

## CRT sweep logging (v3.60.0) — RECORDED, never acted on

**The problem it addresses.** Selection is not the issue; TIMING is. Live,
2026-09-19:

    79.1% of entries are ALREADY LOSING when the guardian first sees them
    entered green -> n= 9, mean net ROI +3.60, win 55.6%
    entered red   -> n=34, mean net ROI -3.37, win 26.5%

and NOTHING in the ~40-field entry context separates them — every field's
median split came in under 1 ROI point, none monotonic. So no screening knob
can fix this. The TRIGGER has to change.

Today's trigger is distance (`AUTO_CALLBACK_*` retrace), which says nothing
about whether the move is over. CRT's is structural: a push beyond a level
that FAILED, confirmed by a close back inside. The `wait_*` columns already
say the better price is one candle later (81% of the time, median +0.26%);
CRT is a rule for WHICH candle.

**Which CRT.** The counter-trend form — sweep of the range HIGH, close back
inside, short. That fits a bot that fades strength (RSI >= 75 on coins up
8%+). The trend-following CRT+MDM variant buys dips in an uptrend and would
confirm the REVERSE of every trade taken here.

**`bot/crt.py`** is a pure function over closed candles. The range candle is
five 3m candles aggregated into a 15m block, built from the frame the scanner
ALREADY fetched — no extra request. `scan_runner` stamps every row;
`auto_trader` copies the fields into `entry_context`; the CSV export
auto-discovers them as `ctx_crt_*`, so no export change was needed.

**Design decisions that matter for the analysis:**

* `crt_swept` is **None** when there is not enough data, never False —
  "no opinion" and "no sweep" must not look alike.
* Closing BEYOND the level is **False**, not a weaker sweep. That is the
  rule's whole point: continuation is then more likely than reversal.
* Both ends swept records `side="both"` and `swept=False` — CRT says nothing
  about which side wins, so it does not guess.
* Detection cannot raise; junk input returns the empty shape.

**A test asserts `evaluate_candidate` never reads these fields.** A gate that
read them would destroy the measurement it exists to make.

**The question to answer in a week:** do entries that followed a confirmed
sweep start green more often than the 20.9% that currently do? Group the
export on `ctx_crt_agrees` against `roi_at_first_sight`.

**Not validated.** The source is a walk-through of selected examples with no
test in it. This records the condition so the account's own trades can settle
it.

## wallet_gap SOLVED: the fee reserve, valued at COST (v3.61.0)

LSK 2026-09-20 — the only trade on a freshly reset live account — isolated it
completely:

    Binance   Closing PNL -0.22 | Funding 0.00 | Trading Fee -0.04 (-0.0001 BNB)
    bot       realised -0.2273  | fees 0.0395  | net -0.2667
    USDT      86.28317071 -> 86.0559 = -0.2273   (the CLOSING PnL, exactly)

    ACCOUNT RETURN -0.26% = wallet delta / start   (gross, fee invisible)
    bot            -0.31% = net incl. fee / start  (fee counted)
    gap           +$0.04  = the BNB fee, to the cent

Nothing was mis-recorded. The fee left the BNB balance and never touched USDT,
so a USDT-only figure is STRUCTURALLY blind to it.

**`account_value()` on the guardian** returns USDT + the fee reserve, and the
three return-card call sites now read it via `_account_wallet()` in api.py.
Sizing is untouched and still uses `resolve_usdt_balance` — ENTRY_RISK_PCT
means capital that can absorb a LOSS, and the fee reserve cannot take one.

**Valued at COST, not at the mark** (`fee_reserve_basis`). BNB is held to pay
fees, not as a position. At the live mark a $500 reserve moving 10% is
+/-$50 — larger than a day of trading — and would appear as if the bot had
earned it. The basis is recorded the first time a price is seen, persisted in
`asset_basis`, and NEVER re-based: re-basing on a later price reintroduces the
drift this exists to prevent.

That also removes the lookup-failure mode entirely: a basis needs no price
after the first one.

**Both operator scenarios verified by test:**

* **Demo** — no BNB asset, so `account_value == USDT` and the gap is zero by
  construction, no special case.
* **BNB runs out mid-run** — every step is `closing - fee` whether BNB or
  USDT paid it. No discontinuity, because the BNB was already counted before
  it was spent.

**A test caught a bug in this work.** The degraded path returned 0.0 on an
unparseable payload — precisely the "blip reads as a loss" failure the design
exists to prevent. It now falls back to the last good cached balance; every
malformed shape returns 86.28, never zero.

**Watch:** the gap now grows ~$0.04/trade, ~$1.70/day at 43 trades. That is
the reserve draining, not a discrepancy. At $2.86 the runway is ~76 trades.

## v3.61.0 SHIPPED TWO CARD BUGS — fixed in v3.61.1

Both were mine, both were visible on the first deploy, and neither was caught
by 1224 passing tests because both live in how REAL payloads are shaped.

**1. DEMO read ACCOUNT RETURN +100.00%, gap +$5000 on a $5000 wallet.**
`wallet_now` came back as 10000. `resolve_account_value` summed the WHOLE
`assets[]` array, so a payload that lists USDT twice — or carries a total row
beside the per-asset rows — doubled the margin balance, the largest number
there.

Fixed by never summing USDT from the array: `account_value()` now takes USDT
from `resolve_usdt_balance` (the resolver sizing already trusts) and adds only
NON-USDT assets. Doubling is impossible by construction.

**2. LIVE read ACCOUNT RETURN +3.01%, gap +$2.86 on a $2.86 BNB balance.**
The baseline was stored in USDT-only units while the card now reads USDT +
reserve, so the reserve was counted on ONE SIDE ONLY and appeared as a gain
out of nowhere. Changing what `wallet_now` means without changing
`wallet_start` to match.

Fixed three ways: new baselines are recorded from `account_value()`;
`wallet_start_basis` is persisted; and a baseline written before this version
is migrated ONCE on the first balance poll, with a WARNING naming the old and
new figures.

**Also corrected:** an unusable `assets[]` array now reports
`usdt-only (no assets array)` instead of `account_value`, which would have
implied the reserve was measured and found to be zero.

## The BNB discount is close to a coin flip

Separating two things that were run together earlier:

* **The fee** is charged in USDT terms and converted at spot. BNB at 0.8x
  means 25% MORE BNB units for the same dollar cost. Genuinely price-neutral.
* **The unspent reserve** is a BNB position. That loss is real.

Break-even is SCALE-INVARIANT: **a 10% fall in BNB wipes out the entire
discount**, at any reserve size. Both sides scale with the runway — a bigger
reserve earns more only by lasting longer, and carries price risk for exactly
that longer period.

At ~70% annual vol, one sigma over a two-week runway is ~13% — larger than the
prize. Measured rate: $1.70/day in fees, so the discount is $0.17/day.

Note v3.61.0's cost-basis valuation makes a BNB drawdown invisible in the
cards. Correct for attributing BOT performance; it does NOT tell you what the
reserve is worth.

## A REJECTED armed trail cost the position its protection (v3.62.0)

牛来 2026-09-20. Binance's order history:

    09:55:37  Trailing Stop Sell  Activation Price >= 0.1007672   REJECTED

The bot's log, one second apart:

    06:55:38  TRAIL-RESPONSE id=2000001449476714 status=None
    06:55:38  ARMED native trailing stop ... locks in ~+2% ROI
    06:55:39  Cancelled ALGO order ...545 - adaptive trail superseded

**`status=None` is the whole problem.** `_accepted_id` decides acceptance from
`order["status"]`, and the ALGO endpoint never populates it — every
TRAIL-RESPONSE in the logs reads `status=None`. Rejection there is
ASYNCHRONOUS: the call returns 200 with an id, the order is refused
afterwards. So the guard written for exactly this (its docstring names OP and
牛来) has never worked, for the same reason `activationPrice` never worked —
the code assumes `/fapi/v1/order` semantics on an endpoint that does not share
them.

Consequence: live protection was cancelled for an order that did not exist.
The position held NOTHING for 27 seconds, while five profit-floor attempts
were refused with -2021 (peak +7.3%, current negative). It closed +0.69% on
luck.

**The fix gates the CANCELS, not the record.** `_confirm_resting()` checks the
algo book for the id. The trail is recorded either way — a phantom id is
caught by the audit, and discarding a real one because the book lagged a cycle
would mean arming never sticks. Only the supersede, the fixed-stop cancel and
`armed_replaced_stop` wait for positive confirmation. Unconfirmed keeps
everything already protecting the position and retries next cycle.

An unreadable book returns False: being unable to confirm and being rejected
must reach the same safe outcome.

**The fixture had no algo book**, so every confirmation failed and 7 tests
broke — which is itself the point: a fake that cannot model where these orders
live cannot test them. `FakeExchange` now keeps an `_algo` list, serves
`fapiPrivateGetOpenAlgoOrders`, removes cancelled ids, and has
`reject_next_algo` to reproduce the accepted-call/refused-order shape.

Five new tests, including the live scenario end to end.

**Also:** a second fixed-size source window (`src[i:i + 1400]`) broke on a new
comment. Sliced to the block instead. That is the second time this pattern has
failed; prefer slicing to a marker.

## Clearing a halt was a no-op, and the trailing limit was half-built (v3.63.0)

Two defects, both introduced by v3.57.0's trailing change.

**1. `reset_halt` no longer resumed trading.** It rebased
`day_start_balance`, which was right when the limit measured from the OPEN.
Once the limit trailed `day_peak_balance`, the peak survived untouched and the
drawdown from it was still over the limit, so the next cycle halted again —
the exact no-op the method's docstring says it was written to fix.

**2. There are TWO daily-limit implementations and only one was changed.**
`check_safety` (candidate flow) trailed the high; `_check_daily_drawdown`
(independent of candidate flow) still measured from the day's open. So the
trailing limit has been half-applied since v3.57.0. Both now share the
reference, and a test asserts they do.

**The fix separates two concerns that were sharing a field.**
`day_start_balance` is what the TODAY card measures from; moving it to clear a
halt silently restates the day's return. So the halt now carries its OWN
`halt_base_balance` and `halt_peak_balance`, persisted, seeded from the day's
open and reset at rollover. `reset_halt` moves ONLY those:

    Halt cleared. The HALT reference is rebased 91.20/91.20 -> 90.28
    (base/peak) ... The TODAY card is unchanged: it still measures
    from 90.77.

`day_peak_balance` remains as the day's true high for reporting, never moved
by an operator action.

**Verified end to end:** halt fires, reset resumes trading, day baseline and
day peak unchanged, and a further fall halts again from the new reference —
one further allowance, not an exemption.

**A test of mine had wrong arithmetic** (2% off a high against a 5% limit) and
failed for the right reason. The corrected case is the property itself: -4.0%
from the open is INSIDE a 5% limit, -5.9% from the high is outside it.

## ARM-AT-ENTRY NEVER SURVIVED ITS OWN FIRST CYCLE (v3.64.0)

`_cancel_superseded_stops` protects `floor_stop_id` and `adaptive_trail_id`.
It does NOT protect `native_trail_id`. Arm-at-entry places the armed trail at
adoption, the id lands in `_all_stop_ids`, and the very next fixed-stop
placement sweeps it:

    LIVE 2026-09-20   08:03:19 ARMED AT ENTRY id=...7417
                      08:03:23 Placed stop    id=...7515
                      08:03:23 Cancelled ALGO ...7417      (4 seconds)

    DEMO 2026-09-20   07:50:08 ARMED AT ENTRY id=...786
                      07:50:12 Cancelled ALGO ...786       (4 seconds)

Both instances, every position, since arm-at-entry was enabled. So the feature
has never actually protected a trade, and the poll-driven RATCHET has gone on
doing all the work — which is exactly what the operator noticed when asking
why the bot still ratchets now that Binance trails correctly.

**This is the same class as the SOLV case the method's own docstring
records**: a sweep that did not know about a protection added after it was
written. It has now eaten the adaptive trail once and the armed trail once.
A test asserts all three poll-independent protections are in the protected
set, so the next one added cannot repeat it.

**Found by tracing, not reading.** Three attempts to locate it by reading the
code were wrong (prev_order_id, adopt_state, the supersede block). Wrapping
`_cancel_stop` with a stack trace found it in one run.

## The ratchet is redundant once the armed trail survives

Measured on LSK live 2026-09-20: the ratchet fired at +6.64% ROI after FIVE
cancel/replace cycles. A single native trail at the same 3% ROI callback would
have exited at +6.89% — a 0.025% difference, with ONE order and no polling.

The stack exists because a Binance trail has ONE fixed `callbackRate` for its
whole life, and a position needs two widths: wide before it is ahead (so noise
does not close it), tight after. Two trails at different widths is the honest
answer to that. The ratchet was a third answer, from before native trails
worked.

**Minimum defensible set, once v3.64.0 is running:**

* wide adaptive trail from adoption — survives a dead guardian
* tight armed trail, `activatePrice` at the arm level — survives the poll gap
* profit floor — promises a LEVEL, which no trail does

Each ratchet step is a cancel plus a place: five windows where the old stop is
gone and the new one is unconfirmed. Given a placement can return an id and
not exist (牛来), removing the ratchet removes risk as well as churn.

**DONE in v3.65.0.** `GUARD_RATCHET_ENABLED`, default FALSE. The early return
now fires whenever a native trail rests AND a fixed stop already exists — the
second clause preserves v3.56.0's fix, since a trail armed AT ENTRY has
replaced nothing and the initial stop must still be placed.

Verified by running the cycle, not by the suite:

    ratchet=True   STOP_MARKET placed=5  cancels=7
    ratchet=False  STOP_MARKET placed=1  cancels=1

**A regression this introduced, caught the same way.** With the ratchet off,
the supersede sweep cancelled the FIXED STOP while `state.stop_order_id` still
named it — the position lost its stop and the record disagreed with the
exchange. The sweep exists to retry a cancel that failed at ARMING, so it now
runs only when `armed_replaced_stop` is true. Under arm-at-entry the trail
replaced nothing and the two coexist by design.

The full suite passed through both the bug and the fix. Four tests now cover
it.

**This changes every exit**, so it lands mid-measurement. Deliberate: the
armed trail was being swept anyway (see above), so the week's exits were never
going to be comparable with the days before it.

## THE ADAPTIVE TRAIL WAS REJECTED AND NOTHING NOTICED (v3.66.0)

AKE demo 2026-09-20. Binance order history:

    13:18:49  Trailing Stop Sell  activation >= 0.0970140   Finished (the ENTRY)
              Time Triggered: 13:21:11
    13:21:22  Trailing Stop Buy   activation <= 0.1027323   REJECTED
    13:21:24  Trailing Stop Buy   activation <= 0.0973857   Canceled

The bot logged the rejected one as placed — `TRAIL-RESPONSE ... status=None
kept=0.1027323` — because the algo endpoint returns 200 with an id and refuses
the order AFTERWARDS. The adaptive trail is the ONLY loss cap between the fill
and the fixed stop. The position closed at **-119.68% ROI**.

**Why it was rejected:** a BUY trail needs its activation AT OR BELOW the
current price. The adaptive trail sends no activatePrice, so Binance filled in
its own default — `0.1027323` against a mark of `0.102718`, ABOVE it by
0.014%. On a market moving 0.38% of price per second, the default was stale
before it was written.

**`_confirm_resting` already existed** (v3.62.0) and gated only the SUPERSEDE.
The placement that matters most was never checked. It now is.

**The id is recorded even when unconfirmed, deliberately.** Clearing it makes
the next cycle place another trail, and an unreadable algo book then places
one EVERY cycle — 40 orders in a short test run when this was first written.
What changed is that the failure is LOUD and the position is marked
UNPROTECTED, instead of a refused order counting as a loss cap.

## A correction that matters: the guardian was NOT slow

An earlier reading of these logs put the unprotected window at 152 seconds,
measured from ENTRY ORDER PLACED to adopted. That was wrong: entries are
trailing stops that rest until triggered. Binance's `Time Triggered` field
gives the real fill at **13:21:11**, and the guardian adopted at **13:21:21**.

**Ten seconds.** The fast poll (`_has_pending_entries` ->
`GUARDIAN_PENDING_POLL_INTERVAL`) is wired at `main.py:192` and works.

The move was 6.44% of price in ~17 seconds — **0.38%/second**, or 7% of margin
per second at 18.6x. No poll-based protection reaches that, and no reduce-only
order can exist before the fill.

## AUTO_MAX_ATR_PCT — a ceiling, OFF by default

Mirrors `AUTO_MIN_ATR_PCT`. A coin can be too FAST to protect, not only too
quiet to pay for. Default 0 (off) at the operator's decision: the adverse-move
risk is accepted for now, to be enabled once there is data on where the line
sits. A MISSING ATR does not trip the ceiling (the floor refuses on missing;
the ceiling must not).

## The TODAY card's win count is correct; WIN RATE is gross

Not a halt-reset bug — `reset_halt` touches only the halt's own base and peak.
The two cards count differently:

    LSK 2026-09-20 13:46:53  realised +2.1282  fees 2.6698  net -0.5416
    WIN RATE (analysis.py:101) counts _roi()      -> +1.59% -> WIN  -> 1/7 = 14%
    TODAY    (analysis.py:353) counts _realised() -> -0.54  -> LOSS -> 0W

TODAY is net of fees. A "winner" that costs 0.54 is not one, so TODAY is the
honest figure — but the two cards disagreeing is its own problem. NOT changed;
decide whether WIN RATE should also be net.

## THE INCOME LEDGER INHERITED THE PREVIOUS TRADE (v3.67.0)

AVAAI demo 2026-09-20, two trades on the SAME SYMBOL twelve seconds apart:

    12:30:10  trade 1 closed  +35.0812   wallet 4781.97 -> 4815.72  (+33.75)
    12:30:22  trade 2 sized
    12:36:48  trade 2 closed  "+32.5135" wallet 4815.72 -> 4810.42  ( -5.30)

Binance's own closing order for trade 2: **Total PNL -2.56773**, fee 1.36346.

The bot had all three numbers and chose the wrong one. Its own log:

    exchange realisedPnl -2.5677 disagrees in sign with the computed +4.3266
      -> using the computed value
    computed P&L +4.3266 disagrees with the income ledger +32.5135
      -> using the ledger

`INCOME_LOOKBACK_PAD_S = 120` exists because the entry commission is charged
at the FILL, before the first poll — starting the query later halved every fee
figure. But a re-entry inside that window makes the query sum the PREVIOUS
trade's rows into this one. Twelve seconds apart, 120-second pad.

**Fix:** `_last_close_ms[symbol]` is stamped at every close and the income
window is clamped to `prev + 1`. It only ever TIGHTENS the window — an older
close cannot widen it — so the pad still does its job on a first entry.

**This corrupts history as well as cards.** Any symbol re-entered within 120s
has an inflated realised figure in the journal. Consider a reset before the
measurement week if AVAAI-style rapid re-entries are common.

## WIN RATE is now net of fees

LSK 2026-09-20 13:46:53: realised +2.1282, fees 2.6698, **net -0.5416**. The
WIN RATE card counted it a win (`_roi` +1.59%); TODAY counted it a loss
(`_realised`). Both were working as written and disagreeing about one trade.

`group_stats` now counts a win on `_realised(t)`, falling back to ROI only
when no net figure exists so older records still score rather than vanishing.
At ~1% of margin per round trip this decides a large share of trades — the
7-trade demo day goes from 14.3% to 0.0%.

## Shadow log: ATOMIC questions, verdict composed in CODE (v3.68.0)

There is no longer a `verdict` question. Asking one composite "would you take
this trade?" hands the weighting to the model, and a change of priorities then
becomes a prompt rewrite whose effect on past rows is unknowable.

jev is now asked FOUR atomic questions — `conviction`, `looks_exhausted`,
`regime_aligned`, `structure_intact` — and `_compose_verdict` assembles
ENTER/SKIP from `_WEIGHTS` and `_ENTER_THRESHOLD`, both in the module.

**Every component is recorded RAW on every row**, with the weights and
threshold used. So the composition can be RE-FITTED offline against outcomes
without re-running a single call, and a row composed under one set of weights
is never silently compared against another.

`confidence` is now the model's WEAKEST component, not an average: a verdict
assembled from four judgements is only as trustworthy as its worst input, and
averaging hides exactly the case worth abstaining on.

**Starting weights are deliberately plain and are NOT a tuned result:**

    not_exhausted 1.0 | regime_aligned 1.0 | structure_intact 0.5 | conviction 0.5
    threshold 0.55

**One case to look at first.** "Fighting the regime" (regime_aligned 0.10,
everything else strong) composes to ENTER at 0.572. That looks wrong for a
trend follower — but this bot deliberately FADES strength, shorting RSI >= 75
on coins up 8%, so being against the regime is often the trade. Whether that
weight is wrong is an empirical question the logged components can answer.

## Container

`typesafe-sdk` is installed in the Dockerfile with `|| true`, NOT pinned in
requirements.txt. The module already disables itself with one warning if the
import or `TYPESAFE_API_KEY` is missing, so a trading container must never
fail to BUILD over an advisory logger. Verify after a rebuild:

    python -c "import typesafe_sdk"

To enable: `TYPESAFE_API_KEY=...`, `SHADOW_ENABLED=true`. DEMO FIRST — live
already has the callback variable in flight, and this adds a network
dependency on every entry decision.

## NOT done, deliberately

- **No threshold changes.** Ship the price fix alone so the next run is
  attributable.
- **`GUARD_STOP_WORKING_TYPE` stays `MARK_PRICE`.** Switching to
  `CONTRACT_PRICE` is an env-only change and would align the trigger with what
  the guardian reads — but only where that number is a *current* price. Two of
  the four were stale (OP's sits four minutes back on the chart, 牛来's below
  the hour's lowest print), and no trigger reference makes a stale number
  current. Worse, those orders would then be *accepted* at levels unrelated to
  the market: refused is loud, accepted-and-wrong is silent. The -2021s are
  currently the only detector this fault has.
- **Trigger price vs fill price** still not recorded. Outstanding since the
  previous handover.

---

## Open, in evidence order

1. **ONE 17:00 does not fit the pattern.** Its +31.8% reconciles with a
   plausible entry of 0.0016390 on the chart, its trail was **accepted** and
   Finished, and it still gave back 47 ROI points on a 3% callback. Looks like
   a genuine peak and a trail firing far from where its callback says.
   Different fault; unresolved.

   **Instrumented for the next occurrence** (v3.42.0), because the record had
   neither of the two facts needed to diagnose it:

   - `TRAIL-ACTIVATION` on every trail placement — the activation price the
     exchange assigned (read back from `info.activatePrice`; we send none, so
     Binance derives it), the mark at that moment, the gap, and whether the
     trail is LIVE or DORMANT. A BUY trail closes a short and activates at
     price <= activation, so an activation below mark protects nothing until
     price moves further in profit. That case gets its own WARNING.
   - `GIVE-BACK` at close — peak, final, the points surrendered, the callback
     they should have been bounded by, every resting order id, and the
     reconstructed exit price. A give-back materially larger than the callback
     can explain is logged at WARNING with "MORE THAN THE TRAIL CAN EXPLAIN"
     rather than sitting at INFO.

   Wrapped so it can never take the close path down with it.
2. **Where the stale prices come from.** `previousClose` was tested and
   refuted — it would have given OP +139% against a logged +9.8%. The new
   source logging should name it in one trade.
3. **`GUARD_BREAKEVEN_STOP_ROI=2` sits below the fee load** (~3.55% of margin
   in ROI terms). COTI closed +1.46% ROI and **-0.008 USDT**. Binance's
   `callbackRate` minimum of 0.1% is 2% ROI at 20x, so the trail cannot be
   tighter either. Settle first whether "never close negative" means Binance's
   displayed ROI or the wallet.
4. **`GUARD_STOP_WORKING_TYPE` is not validated.** A typo in the compose env is
   `.upper()`-ed and passed to Binance. `GuardConfig.validate()` does not
   constrain it.
5. **Entry give-back.** Entries are trailing-stop orders; ONE's carried a 2.31%
   callback, 牛来's filled 1.08% below its activation. At 20x that is tens of
   ROI points surrendered before the position starts.

---

## What to watch on the next run

- **`PEAK` lines** — the price and `src` on any implausible peak. This is the
  measurement the whole diagnosis was missing.
- **`NOT a live price`** — if this fires, the stale source is named.
- **`PRICE-DIVERGENCE`** — size and frequency of the mark/last gap.
- **`was REJECTED by the exchange`** — how often placements were silently
  failing before.
- **-2021 refusals should become rare.** If they do not, the gap is larger
  than 0.1% of price and the floor distance itself needs to change.
- Rate-limit codes (-1003, 429, 418) — the premium index adds a call per
  symbol per poll, cached 1s.

---

## Method — keep to it

Measure before prescribing. This session produced four wrong explanations in a
row — mark/last divergence, a dormant trail, `previousClose`, a stale ticker —
and each was refuted by the operator's next screenshot. What survived was only
what Binance's own records showed. The order history and the activation prices
were worth more than every inference drawn without them.

Two traps hit here:

- **A string-replace edit that does not match makes no change and reports no
  error.** Assert the target matched AND the result is present.
- **Comments can encode false premises and get enshrined in tests.** A test
  asserted `"activationPrice" not in params` on the strength of a comment
  claiming Binance activates at the mark. It does not.

---

## Environment

Live runs 10x, demo 20x. Logs are UTC; the UI and the Binance app render
local (operator is UTC+3), so subtract three hours from a screenshot time.
`GUARD_TRAIL_CALLBACK_ROI=3` is set in compose — note this is **not**
`GUARD_CALLBACK_ROI`, which defaults to 10 and feeds `desired_stop_roi`.

Version lives in `bot/__init__.py`, surfaced at `bot/api.py:539`.

Nothing goes live until demo shows the refusals stop and no phantom peaks.
