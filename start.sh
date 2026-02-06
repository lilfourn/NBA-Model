#!/bin/sh
set -e

echo "=== start.sh: Running database migrations ==="
alembic upgrade head
echo "=== start.sh: Migrations complete ==="

echo "=== start.sh: Verifying app imports ==="
python -c "from app.main import app; print('App imported OK')"

echo "=== start.sh: Starting gunicorn on port ${PORT:-8000} ==="
exec gunicorn app.main:app --config gunicorn_conf.py
