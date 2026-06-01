FROM python:3.12-slim

# Enable BuildKit inline cache — embeds cache metadata in pushed image
# so subsequent builds can use it as a cache source
ARG BUILDKIT_INLINE_CACHE=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create logs dir
RUN mkdir -p logs

# Non-root user for security
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

CMD ["python", "main.py"]
