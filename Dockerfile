# Base pinned by digest for reproducible releases. The human-readable tag was
# python:3.12-slim (resolved 2026-09-16).
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Build identity: the image has no .git checkout, so the revision is injected
# here and surfaced by /api/version. Neutral source label (no host-specific
# URL); override both at build time with
#   docker build --build-arg REVISION=$(git rev-parse --short HEAD) .
ARG REVISION=unknown
LABEL org.opencontainers.image.revision=$REVISION \
      org.opencontainers.image.source="https://github.com/thenaxon/somora_realtimevoice" \
      org.opencontainers.image.title="somora_realtimevoice" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REALTIMEVOICE_DATA_DIR=/app/data
ENV REALTIMEVOICE_REVISION=$REVISION

WORKDIR /app

# ffmpeg decodes speaking models that answer with MP3
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin rtv \
    && mkdir -p /app/data \
    && chown -R rtv:rtv /app
USER rtv

EXPOSE 8000

# /api/health is the unauthenticated liveness path; no curl in slim, so use
# the interpreter's stdlib. Any HTTP 200 on 127.0.0.1 is healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"]

CMD ["python", "-m", "realtimevoice", "--host", "0.0.0.0", "--port", "8000"]
