FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "gunicorn>=22,<24"

COPY --chown=app:app . .

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/', timeout=4).status == 200 else 1)"

# gthread keeps NDJSON progress streaming working; timeout is disabled because a
# single assessment (OCR + grading + retries) can legitimately run for a long time.
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "1", "--threads", "4", \
     "--timeout", "0", "--graceful-timeout", "30", \
     "--access-logfile", "-", "--error-logfile", "-", "app:app"]
