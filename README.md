# Binance Momentum Scalping Bot

Automated spot scalper targeting coins in steady upward trends using technical confirmation before entry and strict risk controls on exit.

## Strategy overview

**Entry conditions (all must pass):**
| Signal | Threshold | Purpose |
|---|---|---|
| EMA20 > EMA50 | — | Uptrend structure |
| ADX | > 25 | Trend has strength, not ranging |
| RSI | 50–65 | Healthy momentum, not overbought |
| +DI | > 20 | Bullish directional pressure |
| Close | > VWAP | Buying above fair value |

**Exit logic:**
- Trailing stop: lifts with price, triggers on `TRAILING_STOP_PCT` drawdown from peak
- Take profit: closes at `TAKE_PROFIT_PCT` gain from entry
- Timeout: closes after `MAX_HOLD_CANDLES` candles regardless

**Risk per trade:** ATR-scaled, capped at `MAX_PORTFOLIO_PCT` of free balance

---

## Quick start

### 1. Get Binance API keys

**Testnet (start here):**
1. Go to https://testnet.binance.vision
2. Log in with GitHub
3. Generate API key + secret
4. Fund testnet account (free fake USDT available on the page)

**Live (after validated on testnet):**
1. Binance → Profile → API Management
2. Enable: Spot & Margin trading. Disable: Withdrawals.
3. Whitelist your VPS IP address.

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
# Keep TESTNET=true until you're satisfied with live behavior
```

### 3. Run locally

```bash
pip install -r requirements.txt
mkdir -p logs
python main.py
```

### 4. Run via Docker

```bash
docker compose up -d
docker compose logs -f bot
```

---

## Deploying to DigitalOcean VPS via GitHub Actions

### One-time VPS setup

```bash
# On your VPS
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
mkdir -p ~/scalping-bot

# Copy your .env file to the VPS (do this once manually)
scp .env user@your-vps-ip:~/scalping-bot/.env
```

### GitHub secrets required

Go to your repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|---|---|
| `VPS_HOST` | Your DigitalOcean droplet IP |
| `VPS_USER` | SSH username (e.g. `root` or `ubuntu`) |
| `VPS_SSH_KEY` | Contents of your private SSH key |
| `GHCR_TOKEN` | GitHub personal access token with `read:packages` |

### Deploying

Push to `main` → GitHub Actions automatically:
1. Runs tests
2. Builds and pushes Docker image to GHCR
3. SSHes into your VPS and restarts the bot with zero downtime

To deploy manually: Actions → "Deploy Scalping Bot" → Run workflow.

---

## Switching testnet → live

1. On your VPS, edit `~/scalping-bot/.env`:
   ```
   TESTNET=false
   BINANCE_API_KEY=your_live_key
   BINANCE_API_SECRET=your_live_secret
   ```
2. Restart: `cd ~/scalping-bot && docker compose up -d`

No code change, no redeploy needed — it's purely config.

---

## Strategy profiles

Two profiles are provided in `.env.example`. Copy the block you want into your `.env`.

### Profile A — Conservative (default)

High-liquidity coins only (BTC, ETH, SOL, BNB, XRP). Stricter trend confirmation means fewer but higher-quality entries. Best for preserving capital while generating steady small gains.

| Parameter | Value | Reasoning |
|---|---|---|
| `MIN_VOLUME_USDT` | 5,000,000 | Top-tier liquid pairs only — deep order books, low slippage |
| `MAX_SPREAD_PCT` | 0.08% | Tight spread — minimal entry/exit cost |
| `ADX_MIN` | 28 | Trend must be well established before entering |
| `RSI_MIN` / `RSI_MAX` | 52–62 | Momentum confirmed, not overbought — avoids late entries |
| `TRAILING_STOP_PCT` | 0.8% | Tight trail — locks in gains quickly on liquid coins |
| `TAKE_PROFIT_PCT` | 1.2% | Realistic target for coins that move in smaller increments |
| `MAX_OPEN_POSITIONS` | 3 | Concentrated — only high-conviction trades |
| `MAX_HOLD_CANDLES` | 12 | 1hr on 5m — exit if trade goes nowhere |
| `RISK_PER_TRADE_PCT` | 1.0% | 1% of balance at risk per trade |
| `MAX_PORTFOLIO_PCT` | 25% | Max 25% of balance in any single position |

### Profile B — Mid-cap momentum

Opens the universe to mid-caps (XLM, INJ, STG and similar). Compensates for lower liquidity with stricter trend filters and wider stops. Best for larger gains per trade, at the cost of higher individual coin risk.

| Parameter | Value | Reasoning |
|---|---|---|
| `MIN_VOLUME_USDT` | 1,000,000 | Opens mid-caps while excluding true micro-caps |
| `MAX_SPREAD_PCT` | 0.15% | Tolerates slightly thinner order books |
| `ADX_MIN` | 30 | Stricter — mid-caps need stronger trend confirmation |
| `RSI_MIN` / `RSI_MAX` | 52–62 | Same window — avoids chasing pumps |
| `TRAILING_STOP_PCT` | 1.2% | Wider trail — mid-caps are noisier, tight stops get clipped |
| `TAKE_PROFIT_PCT` | 2.0% | Higher target justified — mid-caps move further when they move |
| `MAX_OPEN_POSITIONS` | 3 | Same — don't spread thinner just because the universe is wider |
| `MAX_HOLD_CANDLES` | 16 | 80min — mid-caps can take longer to reach target |
| `RISK_PER_TRADE_PCT` | 0.8% | Slightly less per trade — offsets higher individual coin risk |
| `MAX_PORTFOLIO_PCT` | 20% | Tighter cap — mid-caps can gap down harder |

**On pump-and-dump coins:** Anything showing >15% 24h gain will typically have RSI above 70 and fail the `RSI_MAX` filter regardless of profile. If you ever see the bot entering a coin that has already pumped hard, tighten `RSI_MAX` to 60.

**Switching profiles** requires only a `.env` change and a container restart — no rebuild needed:

```bash
# Edit .env, then:
docker compose restart bot
```

---

## Logs

```bash
# Live logs from VPS
docker compose logs -f bot

# Or check the file
tail -f logs/bot.log
```

---

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past strategy performance does not guarantee future results. Never trade with money you cannot afford to lose.
