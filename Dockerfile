FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Create a non-root user
RUN addgroup --system app && adduser --system --ingroup app app

FROM base AS deps
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM base AS dev
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt -r requirements-dev.txt
ENV PATH="/opt/venv/bin:$PATH"
COPY . .
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS prod
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini gunicorn_conf.py start.sh ./
COPY data/name_overrides.json data/team_abbrev_overrides.json ./data/
RUN chmod +x start.sh
USER app
EXPOSE 8000
CMD ["./start.sh"]
