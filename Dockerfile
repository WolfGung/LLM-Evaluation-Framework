FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 toolshop

# Dependencies first, from the package metadata only, so a code change does
# not reinstall them.
COPY pyproject.toml README.md LICENSE ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# .dockerignore is an allowlist: only the code, config/ and cassettes/ come in,
# never .env or local files. Compose mounts ./cassettes over the copy, so new
# recordings are served without a rebuild.
COPY . .
RUN pip install --no-deps . \
    && rm -rf build *.egg-info \
    && mkdir -p cassettes

USER toolshop

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
