FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# .dockerignore is an allowlist: only the code, config/ and cassettes/ come in,
# never .env or local files.
COPY . .
RUN pip install . \
    && rm -rf build *.egg-info \
    && mkdir -p cassettes \
    && useradd --create-home --uid 10001 toolshop

USER toolshop

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
