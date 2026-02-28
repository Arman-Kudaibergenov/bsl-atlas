# syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/

# Create data directories
RUN mkdir -p /data/source /data/chroma_db

# Environment variables with defaults
ENV EMBEDDING_PROVIDER=openai \
    SOURCE_PATH=/data/source \
    CHROMA_PATH=/data/chroma_db \
    AUTO_INDEX=true \
    HOST=0.0.0.0 \
    PORT=8000 \
    CHUNK_SIZE=1000 \
    CHUNK_OVERLAP=100 \
    MAX_BATCH_SIZE=100 \
    DEFAULT_SEARCH_LIMIT=10

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "src.main"]
