from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NBA Stats API"
    environment: str = "development"
    log_level: str = "INFO"
    cors_allow_origins: str = Field(default="http://localhost:3000,http://localhost:3001", validation_alias="CORS_ALLOW_ORIGINS")
    prizepicks_api_url: str = Field(
        default="http://partner-api.prizepicks.com",
        validation_alias="PRIZEPICKS_API_URL",
    )
    prizepicks_league_id: int = Field(default=7, validation_alias="LEAGUE_ID")
    prizepicks_per_page: int = Field(default=1000, validation_alias="PRIZEPICKS_PER_PAGE")
    prizepicks_timeout_seconds: int = Field(default=20, validation_alias="PRIZEPICKS_TIMEOUT_SECONDS")
    prizepicks_impersonate: str = Field(default="chrome", validation_alias="PRIZEPICKS_IMPERSONATE")
    prizepicks_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        validation_alias="PRIZEPICKS_USER_AGENT",
    )
    nba_stats_api_url: str = Field(
        default="https://stats.nba.com/stats",
        validation_alias="NBA_STATS_API_URL",
    )
    nba_stats_timeout_seconds: int = Field(
        default=20,
        validation_alias="NBA_STATS_TIMEOUT_SECONDS",
    )
    nba_stats_max_retries: int = Field(default=3, validation_alias="NBA_STATS_MAX_RETRIES")
    nba_stats_backoff_seconds: float = Field(default=1.5, validation_alias="NBA_STATS_BACKOFF_SECONDS")
    nba_stats_impersonate: str = Field(default="chrome", validation_alias="NBA_STATS_IMPERSONATE")
    nba_stats_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        validation_alias="NBA_STATS_USER_AGENT",
    )
    nba_stats_origin: str = Field(
        default="https://www.nba.com",
        validation_alias="NBA_STATS_ORIGIN",
    )
    nba_stats_referer: str = Field(
        default="https://www.nba.com/stats/players/boxscores-traditional",
        validation_alias="NBA_STATS_REFERER",
    )
    nba_stats_proxy: str | None = Field(default=None, validation_alias="NBA_STATS_PROXY")
    basketball_reference_base_url: str = Field(
        default="https://www.basketball-reference.com",
        validation_alias="BASKETBALL_REFERENCE_BASE_URL",
    )
    basketball_reference_timeout_seconds: int = Field(
        default=20,
        validation_alias="BASKETBALL_REFERENCE_TIMEOUT_SECONDS",
    )
    basketball_reference_max_retries: int = Field(
        default=2,
        validation_alias="BASKETBALL_REFERENCE_MAX_RETRIES",
    )
    basketball_reference_backoff_seconds: float = Field(
        default=2.0,
        validation_alias="BASKETBALL_REFERENCE_BACKOFF_SECONDS",
    )
    basketball_reference_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        validation_alias="BASKETBALL_REFERENCE_USER_AGENT",
    )
    basketball_reference_proxy: str | None = Field(
        default=None,
        validation_alias="BASKETBALL_REFERENCE_PROXY",
    )
    statmuse_base_url: str = Field(
        default="https://www.statmuse.com",
        validation_alias="STATMUSE_BASE_URL",
    )
    statmuse_timeout_seconds: int = Field(default=20, validation_alias="STATMUSE_TIMEOUT_SECONDS")
    statmuse_max_retries: int = Field(default=2, validation_alias="STATMUSE_MAX_RETRIES")
    statmuse_backoff_seconds: float = Field(default=1.5, validation_alias="STATMUSE_BACKOFF_SECONDS")
    statmuse_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        validation_alias="STATMUSE_USER_AGENT",
    )
    statmuse_proxy: str | None = Field(default=None, validation_alias="STATMUSE_PROXY")
    player_name_overrides_path: str = Field(
        default="data/name_overrides.json",
        validation_alias="PLAYER_NAME_OVERRIDES_PATH",
    )
    team_abbrev_overrides_path: str = Field(
        default="data/team_abbrev_overrides.json",
        validation_alias="TEAM_ABBREV_OVERRIDES_PATH",
    )
    # Cache settings
    cache_dir: str = Field(default="data/cache", validation_alias="CACHE_DIR")
    cache_default_ttl_seconds: float = Field(default=3600.0, validation_alias="CACHE_DEFAULT_TTL_SECONDS")
    cache_nba_stats_ttl_seconds: float = Field(default=7200.0, validation_alias="CACHE_NBA_STATS_TTL_SECONDS")
    cache_prizepicks_ttl_seconds: float = Field(default=300.0, validation_alias="CACHE_PRIZEPICKS_TTL_SECONDS")
    cache_bref_ttl_seconds: float = Field(default=86400.0, validation_alias="CACHE_BREF_TTL_SECONDS")
    cache_statmuse_ttl_seconds: float = Field(default=86400.0, validation_alias="CACHE_STATMUSE_TTL_SECONDS")

    # Collection logging
    collection_log_path: str = Field(default="logs/collection.jsonl", validation_alias="COLLECTION_LOG_PATH")

    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")

    # Model artifact source: "fs" for local filesystem (default), "db" for Postgres
    model_source: str = Field(default="fs", validation_alias="MODEL_SOURCE")
    models_dir: str = Field(default="models", validation_alias="MODELS_DIR")
    ensemble_weights_path: str = Field(
        default="models/ensemble_weights.json",
        validation_alias="ENSEMBLE_WEIGHTS_PATH",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
