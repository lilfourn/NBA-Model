from __future__ import annotations

from sqlalchemy import Boolean, Column, Date, DateTime, ForeignKey, Index, Integer, MetaData, Numeric, Text
from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

snapshots = Table(
    "snapshots",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("league_id", Text, nullable=False),
    Column("per_page", Integer, nullable=True),
    Column("source_url", Text, nullable=True),
    Column("snapshot_path", Text, nullable=True),
    Column("data_count", Integer, nullable=True),
    Column("included_count", Integer, nullable=True),
    Column("links", JSONB, nullable=True),
    Column("meta", JSONB, nullable=True),
)

snapshot_audits = Table(
    "snapshot_audits",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "snapshot_id",
        UUID(as_uuid=True),
        ForeignKey("snapshots.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("summary", JSONB, nullable=False),
)

players = Table(
    "players",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("name_key", Text, nullable=True),
    Column("display_name", Text, nullable=True),
    Column("team", Text, nullable=True),
    Column("team_name", Text, nullable=True),
    Column("position", Text, nullable=True),
    Column("market", Text, nullable=True),
    Column("jersey_number", Text, nullable=True),
    Column("image_url", Text, nullable=True),
    Column("league_id", Text, nullable=True),
    Column("league", Text, nullable=True),
    Column("combo", Boolean, nullable=True),
    Column("team_id", Text, nullable=True),
    Column("attributes", JSONB, nullable=True),
    Column("relationships", JSONB, nullable=True),
)

teams = Table(
    "teams",
    metadata,
    Column("id", Text, primary_key=True),
    Column("abbreviation", Text, nullable=True),
    Column("market", Text, nullable=True),
    Column("name", Text, nullable=True),
    Column("primary_color", Text, nullable=True),
    Column("secondary_color", Text, nullable=True),
    Column("tertiary_color", Text, nullable=True),
    Column("attributes", JSONB, nullable=True),
)

stat_types = Table(
    "stat_types",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("rank", Integer, nullable=True),
    Column("lfg_ignored_leagues", JSONB, nullable=True),
    Column("attributes", JSONB, nullable=True),
)

games = Table(
    "games",
    metadata,
    Column("id", Text, primary_key=True),
    Column("start_time", DateTime(timezone=True), nullable=True),
    Column("end_time", DateTime(timezone=True), nullable=True),
    Column("status", Text, nullable=True),
    Column("is_live", Boolean, nullable=True),
    Column("external_game_id", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=True),
    Column("metadata", JSONB, nullable=True),
    Column("home_team_id", Text, nullable=True),
    Column("away_team_id", Text, nullable=True),
    Column("attributes", JSONB, nullable=True),
    Column("relationships", JSONB, nullable=True),
)

player_game_logs = Table(
    "player_game_logs",
    metadata,
    Column("id", Text, primary_key=True),
    Column("player_id", Text, nullable=True),
    Column("player_name", Text, nullable=True),
    Column("game_id", Text, nullable=True),
    Column("game_date", Date, nullable=True),
    Column("team_id", Text, nullable=True),
    Column("team_abbreviation", Text, nullable=True),
    Column("season", Text, nullable=True),
    Column("season_type", Text, nullable=True),
    Column("stats", JSONB, nullable=True),
)

projection_types = Table(
    "projection_types",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("attributes", JSONB, nullable=True),
)

leagues = Table(
    "leagues",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("rank", Integer, nullable=True),
    Column("active", Boolean, nullable=True),
    Column("projections_count", Integer, nullable=True),
    Column("icon", Text, nullable=True),
    Column("image_url", Text, nullable=True),
    Column("parent_id", Text, nullable=True),
    Column("parent_name", Text, nullable=True),
    Column("f2p_enabled", Boolean, nullable=True),
    Column("has_live_projections", Boolean, nullable=True),
    Column("last_five_games_enabled", Boolean, nullable=True),
    Column("league_icon_id", Text, nullable=True),
    Column("show_trending", Boolean, nullable=True),
    Column("attributes", JSONB, nullable=True),
    Column("relationships", JSONB, nullable=True),
)

durations = Table(
    "durations",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("attributes", JSONB, nullable=True),
)

projections = Table(
    "projections",
    metadata,
    Column("snapshot_id", UUID(as_uuid=True), ForeignKey("snapshots.id", ondelete="CASCADE"), primary_key=True),
    Column("projection_id", Text, primary_key=True),
    Column("league_id", Text, nullable=True),
    Column("player_id", Text, nullable=True),
    Column("stat_type_id", Text, nullable=True),
    Column("projection_type_id", Text, nullable=True),
    Column("game_id", Text, nullable=True),
    Column("duration_id", Text, nullable=True),
    Column("line_score", Numeric, nullable=True),
    Column("line_score_prev", Numeric, nullable=True),
    Column("line_score_delta", Numeric, nullable=True),
    Column("line_movement", Text, nullable=True),
    Column("adjusted_odds", Numeric, nullable=True),
    Column("discount_percentage", Numeric, nullable=True),
    Column("flash_sale_line_score", Numeric, nullable=True),
    Column("odds_type", Integer, nullable=True),
    Column("rank", Integer, nullable=True),
    Column("trending_count", Integer, nullable=True),
    Column("status", Text, nullable=True),
    Column("stat_type", Text, nullable=True),
    Column("stat_display_name", Text, nullable=True),
    Column("projection_type", Text, nullable=True),
    Column("description", Text, nullable=True),
    Column("event_type", Text, nullable=True),
    Column("group_key", Text, nullable=True),
    Column("tv_channel", Text, nullable=True),
    Column("start_time", DateTime(timezone=True), nullable=True),
    Column("board_time", DateTime(timezone=True), nullable=True),
    Column("end_time", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=True),
    Column("is_promo", Boolean, nullable=True),
    Column("is_live", Boolean, nullable=True),
    Column("is_live_scored", Boolean, nullable=True),
    Column("in_game", Boolean, nullable=True),
    Column("today", Boolean, nullable=True),
    Column("refundable", Boolean, nullable=True),
    Column("attributes", JSONB, nullable=True),
    Column("relationships", JSONB, nullable=True),
)

projection_features = Table(
    "projection_features",
    metadata,
    Column(
        "snapshot_id",
        UUID(as_uuid=True),
        ForeignKey("snapshots.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("projection_id", Text, primary_key=True),
    Column("league_id", Text, nullable=True),
    Column("player_id", Text, nullable=True),
    Column("team_id", Text, nullable=True),
    Column("stat_type_id", Text, nullable=True),
    Column("projection_type_id", Text, nullable=True),
    Column("game_id", Text, nullable=True),
    Column("duration_id", Text, nullable=True),
    Column("line_score", Numeric, nullable=True),
    Column("line_score_prev", Numeric, nullable=True),
    Column("line_score_delta", Numeric, nullable=True),
    Column("line_movement", Text, nullable=True),
    Column("stat_type", Text, nullable=True),
    Column("projection_type", Text, nullable=True),
    Column("odds_type", Integer, nullable=True),
    Column("trending_count", Integer, nullable=True),
    Column("is_promo", Boolean, nullable=True),
    Column("is_live", Boolean, nullable=True),
    Column("in_game", Boolean, nullable=True),
    Column("today", Boolean, nullable=True),
    Column("start_time", DateTime(timezone=True), nullable=True),
    Column("board_time", DateTime(timezone=True), nullable=True),
    Column("end_time", DateTime(timezone=True), nullable=True),
    Column("fetched_at", DateTime(timezone=True), nullable=True),
    Column("minutes_to_start", Integer, nullable=True),
)

nba_players = Table(
    "nba_players",
    metadata,
    Column("id", Text, primary_key=True),
    Column("full_name", Text, nullable=True),
    Column("name_key", Text, nullable=True),
    Column("team_id", Text, nullable=True),
    Column("team_abbreviation", Text, nullable=True),
)

nba_games = Table(
    "nba_games",
    metadata,
    Column("id", Text, primary_key=True),
    Column("game_date", Date, nullable=True),
    Column("status_text", Text, nullable=True),
    Column("home_team_id", Text, nullable=True),
    Column("away_team_id", Text, nullable=True),
    Column("home_team_abbreviation", Text, nullable=True),
    Column("away_team_abbreviation", Text, nullable=True),
)

nba_player_game_stats = Table(
    "nba_player_game_stats",
    metadata,
    Column("game_id", Text, primary_key=True),
    Column("player_id", Text, primary_key=True),
    Column("team_id", Text, nullable=True),
    Column("team_abbreviation", Text, nullable=True),
    Column("minutes", Numeric, nullable=True),
    Column("points", Integer, nullable=True),
    Column("rebounds", Integer, nullable=True),
    Column("assists", Integer, nullable=True),
    Column("steals", Integer, nullable=True),
    Column("blocks", Integer, nullable=True),
    Column("turnovers", Integer, nullable=True),
    Column("fg3m", Integer, nullable=True),
    Column("fg3a", Integer, nullable=True),
    Column("fg3_pct", Numeric, nullable=True),
    Column("fgm", Integer, nullable=True),
    Column("fga", Integer, nullable=True),
    Column("fg_pct", Numeric, nullable=True),
    Column("ftm", Integer, nullable=True),
    Column("fta", Integer, nullable=True),
    Column("ft_pct", Numeric, nullable=True),
    Column("plus_minus", Numeric, nullable=True),
    Column("stats_json", JSONB, nullable=True),
)

model_runs = Table(
    "model_runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model_name", Text, nullable=False),
    Column("train_rows", Integer, nullable=False),
    Column("metrics", JSONB, nullable=False),
    Column("params", JSONB, nullable=True),
    Column("artifact_path", Text, nullable=True),
)

projection_predictions = Table(
    "projection_predictions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("snapshot_id", UUID(as_uuid=True), ForeignKey("snapshots.id", ondelete="CASCADE")),
    Column("projection_id", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("pick", Text, nullable=False),
    Column("prob_over", Numeric, nullable=True),
    Column("confidence", Numeric, nullable=True),
    Column("mean", Numeric, nullable=True),
    Column("std", Numeric, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=True),
    Column("details", JSONB, nullable=True),
)

Index("idx_snapshots_fetched_at", snapshots.c.fetched_at)
Index("idx_snapshots_league_id", snapshots.c.league_id)
Index(
    "uq_snapshots_snapshot_path",
    snapshots.c.snapshot_path,
    unique=True,
    postgresql_where=snapshots.c.snapshot_path.isnot(None),
)
Index("idx_snapshot_audits_snapshot_id", snapshot_audits.c.snapshot_id, unique=True)

Index("idx_projections_projection_id", projections.c.projection_id)
Index("idx_projections_player_id", projections.c.player_id)
Index("idx_projections_stat_type_id", projections.c.stat_type_id)
Index("idx_projections_game_id", projections.c.game_id)
Index("idx_projections_league_id", projections.c.league_id)
Index("idx_projections_start_time", projections.c.start_time)

Index("idx_player_game_logs_player_id", player_game_logs.c.player_id)
Index("idx_player_game_logs_game_date", player_game_logs.c.game_date)
Index("idx_prediction_snapshot_id", projection_predictions.c.snapshot_id)
Index("idx_players_name_key", players.c.name_key)
Index("idx_projection_features_snapshot_id", projection_features.c.snapshot_id)
Index("idx_projection_features_player_id", projection_features.c.player_id)
Index("idx_projection_features_game_id", projection_features.c.game_id)
Index("idx_nba_players_name_key", nba_players.c.name_key)
Index("idx_nba_player_stats_game_id", nba_player_game_stats.c.game_id)
