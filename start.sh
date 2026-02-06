#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting gunicorn..."
exec gunicorn app.main:app --config gunicorn_conf.py
