# syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (build-essential for tree-sitter-bsl compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir .

# Compile tree-sitter-bsl grammar into a shared library via gcc.
# Upstream restructured: the generated parser now lives under grammars/bsl/src
# (older layout had it at src/). Locate parser.c dynamically to survive both.
RUN git clone --depth 1 https://github.com/alkoleft/tree-sitter-bsl /tmp/tree-sitter-bsl \
    && mkdir -p /app/lib \
    && PARSER=$(find /tmp/tree-sitter-bsl -path '*/bsl/src/parser.c' -o -path '/tmp/tree-sitter-bsl/src/parser.c' | head -1) \
    && SRCDIR=$(dirname "$PARSER") \
    && echo "Using grammar src: $SRCDIR" \
    && SRCS="$SRCDIR/parser.c" \
    && if [ -f "$SRCDIR/scanner.c" ]; then SRCS="$SRCS $SRCDIR/scanner.c"; fi \
    && gcc -shared -fPIC -Os -I"$SRCDIR" -o /app/lib/bsl.so $SRCS \
    && echo "tree-sitter-bsl compiled OK" \
    && rm -rf /tmp/tree-sitter-bsl

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and compiled grammar from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/lib/bsl.so /app/lib/bsl.so

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

# Health check (python3 is always available in the runtime image, curl is not)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "-m", "src.main"]
