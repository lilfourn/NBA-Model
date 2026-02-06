#!/bin/sh

echo "=== start.sh: Environment check ==="
if [ -z "$DATABASE_URL" ]; then
    echo "WARNING: DATABASE_URL is NOT set"
else
    echo "DATABASE_URL is set (${#DATABASE_URL} chars)"
fi
echo "MODEL_SOURCE=${MODEL_SOURCE:-not set}"
echo "PORT=${PORT:-8000}"

echo "=== start.sh: Running database migrations ==="
alembic upgrade head || echo "WARNING: Migrations failed (may already be up to date)"

echo "=== start.sh: Starting gunicorn on port ${PORT:-8000} ==="
exec gunicorn app.main:app --config gunicorn_conf.py
