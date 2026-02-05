# NBA Stats API

Minimal FastAPI service with a production-grade Docker setup.

## Quick Start (Docker, Dev)

```bash
cp .env.example .env
docker compose up --build
```

Then visit:
- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/metrics/line-movement
- http://localhost:8000/metrics/snapshots

## Production-like Run (Docker)

```bash
docker compose -f docker-compose.prod.yml up --build
```

## Local Run (No Docker)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## PrizePicks Snapshot

Fetch and store an NBA projections snapshot locally:

```bash
python -m scripts.prizepicks.fetch_prizepicks_snapshot
```

Fetch + snapshot + load into Postgres:

```bash
python -m scripts.prizepicks.collect_prizepicks
```

Inspect the latest snapshot structure:

```bash
python -m scripts.prizepicks.inspect_prizepicks_snapshot
```

Normalize the latest snapshot into tables:

```bash
python -m scripts.prizepicks.normalize_prizepicks_snapshot
```

Audit the latest snapshot structure:

```bash
python -m scripts.prizepicks.audit_prizepicks_snapshot
```

Build projection feature rows for the latest snapshot:

```bash
python -m scripts.prizepicks.build_projection_features
```

## Official NBA Game Logs

Fetch official NBA player game logs (used for validation + modeling):

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --date-from 2026-02-01 --date-to 2026-02-03
```

Default date range uses the last 3 completed days (Central time). Override with:

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --range-days 5
```

Fetch league game logs (direct NBA Stats feed):

```bash
python -m scripts.nba.fetch_nba_stats --season 2025-26 --date-from 2026-02-01 --date-to 2026-02-03
```

Fallback sources (per-player) if NBA Stats is blocked:

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --source statmuse --limit 50
```

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --source basketball_reference --limit 50
```

Use a custom player list (newline or PrizePicks JSONL):

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --source statmuse --players-file data/normalized/new_players.jsonl
```

Generate top picks from the latest PrizePicks snapshot:

```bash
python -m scripts.ml.run_top_picks
```

### Player Source Index

Cache per-player URLs for fallback sources (only players in projections):

```bash
python -m scripts.ops.build_player_source_index --season 2025-26
```

Weekly refresh (only missing slugs):

```bash
python -m scripts.ops.refresh_player_sources_weekly
```

### Source Validation

Compare NBA Stats to a fallback source file and report per-stat accuracy:

```bash
python -m scripts.ops.validate_stat_sources --nba-stats-file data/official/nba_player_gamelogs_2025-26_start_2026-02-03.jsonl --fallback-file data/fallback/nba_player_gamelogs_2025-26_start_2026-02-03.jsonl
```

Recommended workflow (guaranteed overlap):

```bash
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --output-dir data/official --range-days 3
python -m scripts.nba.fetch_nba_player_gamelogs --season 2025-26 --source statmuse --output-dir data/fallback --range-days 3 --players-from-nba-file data/official/nba_player_gamelogs_2025-26_2026-01-31_2026-02-02.jsonl --limit 50
python -m scripts.ops.validate_stat_sources --nba-stats-file data/official/nba_player_gamelogs_2025-26_2026-01-31_2026-02-02.jsonl --fallback-file data/fallback/nba_player_gamelogs_2025-26_2026-01-31_2026-02-02.jsonl --date-from 2026-01-31 --date-to 2026-02-02
```

### Player Name Overrides

Create overrides in `data/name_overrides.json` to map PrizePicks names to official sources.

### Team Abbreviation Overrides

If team abbreviations differ between sources, update `data/team_abbrev_overrides.json`.

## Postgres (Neon or Supabase)

Set your connection string in `.env`:

```bash
DATABASE_URL="postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
```

Run migrations:

```bash
alembic upgrade head
```

Load the latest snapshot into Postgres:

```bash
python -m scripts.prizepicks.load_prizepicks_snapshot
```

## Cron (Every 3 Hours)

Add a cron entry on the host to run the collector in Docker:

```bash
0 */3 * * * /Users/lukesmac/nba-stats-project/scripts/cron_collect.sh
```

Daily training run (includes NBA stats refresh):

```bash
0 5 * * * /Users/lukesmac/nba-stats-project/scripts/cron_train.sh
```

Create the log directory once:

```bash
mkdir -p logs
```

Log rotation (delete logs older than 7 days):

```bash
0 0 * * * /usr/bin/find /Users/lukesmac/nba-stats-project/logs -type f -name "*.log" -mtime +7 -delete
```

Email alerts on failures are sent from `scripts/cron_collect.sh` and `scripts/cron_train.sh` using `scripts/ops/send_email.py`.

## Gmail SMTP Alerts

Set the following in `.env` (use a Gmail App Password):

```bash
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your_email@gmail.com"
SMTP_PASSWORD="your_app_password"
SMTP_TO="luke.fournier2023@gmail.com"
```

Test email:

```bash
python -m scripts.ops.send_email --subject "Test: PrizePicks cron email" --body "Test email from cron setup."
```

## ML Baseline

Train a baseline model and store artifacts in `models/`:

```bash
python -m scripts.ml.train_baseline_model
```

Run via Docker:

```bash
docker compose run --rm api python -m scripts.prizepicks.fetch_prizepicks_snapshot
```

```bash
docker compose run --rm api python -m scripts.prizepicks.collect_prizepicks
```

```bash
docker compose run --rm api python -m scripts.prizepicks.inspect_prizepicks_snapshot
```

```bash
docker compose run --rm api python -m scripts.prizepicks.normalize_prizepicks_snapshot
```

```bash
docker compose run --rm api alembic upgrade head
```

```bash
docker compose run --rm api python -m scripts.prizepicks.load_prizepicks_snapshot
```

## Notes
- Dev uses `uvicorn --reload` for live reload.
- Prod uses `gunicorn` with Uvicorn workers and a non-root user.
