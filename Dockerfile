FROM python:3.12-slim

# Enable BuildKit inline cache — embeds cache metadata in pushed image
# so subsequent builds can use it as a cache source
ARG BUILDKIT_INLINE_CACHE=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shadow decision log (bot/shadow_decision.py) — OPTIONAL, and installed
# tolerantly on purpose.
#
# The module already disables itself with one warning if the import or
# TYPESAFE_API_KEY is missing, so a trading container must never fail to BUILD
# over an advisory logger. `|| true` keeps that property: if the package is
# unavailable or renamed, the image still ships and the bot still trades, with
# the shadow log simply off.
#
# Verify after a build with:  python -c "import typesafe_sdk"
RUN pip install --no-cache-dir "typesafe-sdk>=0.1" || \
    echo "typesafe-sdk unavailable — shadow decision log will stay disabled"

# Copy source
COPY . .

# Create logs dir
RUN mkdir -p logs

# Non-root user for security
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

CMD ["python", "main.py"]
