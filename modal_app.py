from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo
from typing import Union

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = Path("/root/project")
REMOTE_STATE_ROOT = Path("/state")

APP_NAME = os.getenv("MODAL_APP_NAME", "nba-stats-project")
SECRET_NAME = os.getenv("MODAL_SECRET_NAME", "nba-stats-env")
STATE_VOLUME_NAME = os.getenv("MODAL_STATE_VOLUME_NAME", "nba-stats-state")
SCHEDULE_TIMEZONE = os.getenv("MODAL_SCHEDULE_TIMEZONE", "America/Chicago")
TRAIN_GPU = os.getenv("MODAL_TRAIN_GPU", "A10")
ENFORCE_DB_SSLMODE = os.getenv("MODAL_ENFORCE_DB_SSLMODE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
SECURE_SSLMODES = {"require", "verify-ca", "verify-full"}

REMOTE_MODELS_DIR = REMOTE_STATE_ROOT / "models"
REMOTE_DATA_DIR = REMOTE_STATE_ROOT / "data"
REMOTE_LOGS_DIR = REMOTE_STATE_ROOT / "logs"
REMOTE_MONITORING_LOG = REMOTE_DATA_DIR / "monitoring" / "prediction_log.csv"
REMOTE_CALIBRATION_DIR = REMOTE_DATA_DIR / "calibration"
REMOTE_HEALTH_REPORT = REMOTE_DATA_DIR / "reports" / "model_health.json"
REMOTE_ABLATION_REPORT = REMOTE_DATA_DIR / "reports" / "ablation_report.json"
REMOTE_CONTEXT_PRIORS = REMOTE_MODELS_DIR / "context_priors.json"
REMOTE_TRAIN_FRESHNESS_REPORT = (
    REMOTE_DATA_DIR / "reports" / "train_data_freshness.json"
)

app = modal.App(APP_NAME)
state_volume = modal.Volume.from_name(STATE_VOLUME_NAME, create_if_missing=True)


def _parse_gpu_spec(raw: str) -> Union[str, list[str]]:
    parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not parts:
        return "A10"
    if len(parts) == 1:
        return parts[0]
    return parts


TRAIN_GPU_SPEC = _parse_gpu_spec(TRAIN_GPU)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .add_local_dir(
        str(REPO_ROOT / "app"), remote_path=str(REMOTE_PROJECT_ROOT / "app"), copy=True
    )
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path=str(REMOTE_PROJECT_ROOT / "scripts"),
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "alembic"),
        remote_path=str(REMOTE_PROJECT_ROOT / "alembic"),
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "data" / "tuning"),
        remote_path=str(REMOTE_PROJECT_ROOT / "data" / "tuning"),
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "name_overrides.json"),
        remote_path=str(REMOTE_PROJECT_ROOT / "data" / "name_overrides.json"),
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "team_abbrev_overrides.json"),
        remote_path=str(REMOTE_PROJECT_ROOT / "data" / "team_abbrev_overrides.json"),
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "stat_expert_routing.json"),
        remote_path=str(REMOTE_PROJECT_ROOT / "data" / "stat_expert_routing.json"),
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "alembic.ini"),
        remote_path=str(REMOTE_PROJECT_ROOT / "alembic.ini"),
        copy=True,
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

shared_kwargs: dict[str, object] = {
    "image": image,
    "volumes": {str(REMOTE_STATE_ROOT): state_volume},
    "secrets": [modal.Secret.from_name(SECRET_NAME)],
}


def _runtime_defaults() -> dict[str, str]:
    return {
        "PYTHONPATH": str(REMOTE_PROJECT_ROOT),
        "CACHE_DIR": str(REMOTE_DATA_DIR / "cache"),
        "TUNING_DIR": str(REMOTE_DATA_DIR / "tuning"),
        "MODELS_DIR": str(REMOTE_MODELS_DIR),
        "ENSEMBLE_WEIGHTS_PATH": str(REMOTE_MODELS_DIR / "ensemble_weights.json"),
        "COLLECTION_LOG_PATH": str(REMOTE_LOGS_DIR / "collection.jsonl"),
        "PLAYER_NAME_OVERRIDES_PATH": str(
            REMOTE_PROJECT_ROOT / "data" / "name_overrides.json"
        ),
        "TEAM_ABBREV_OVERRIDES_PATH": str(
            REMOTE_PROJECT_ROOT / "data" / "team_abbrev_overrides.json"
        ),
    }


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("DATABASE_URL") and env.get("NEON_DATABASE_URL"):
        env["DATABASE_URL"] = env["NEON_DATABASE_URL"]
    for key, value in _runtime_defaults().items():
        env.setdefault(key, value)
    return env


def _apply_runtime_env_defaults() -> None:
    if not os.environ.get("DATABASE_URL") and os.environ.get("NEON_DATABASE_URL"):
        os.environ["DATABASE_URL"] = os.environ["NEON_DATABASE_URL"]
    for key, value in _runtime_defaults().items():
        os.environ.setdefault(key, value)
    project_root = str(REMOTE_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _prepare_state() -> None:
    (REMOTE_DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)
    (REMOTE_DATA_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
    (REMOTE_DATA_DIR / "monitoring").mkdir(parents=True, exist_ok=True)
    (REMOTE_DATA_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (REMOTE_DATA_DIR / "tuning").mkdir(parents=True, exist_ok=True)
    REMOTE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _begin_run(*, verify_database: bool = True) -> None:
    _prepare_state()
    _apply_runtime_env_defaults()
    try:
        state_volume.reload()
    except Exception:
        # If reload is unavailable/busy, continue with current mount state.
        pass
    if verify_database:
        _verify_database_connection()


def _finish_run() -> None:
    state_volume.commit()


def _run_cmd(
    args: list[str],
    *,
    allow_fail: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> int:
    cmd = [sys.executable, *args]
    print(f"[modal] running: {' '.join(cmd)}")
    cmd_env = _runtime_env()
    if env_overrides:
        cmd_env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(
        cmd,
        cwd=str(REMOTE_PROJECT_ROOT),
        env=cmd_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def _verify_database_connection() -> None:
    env = _runtime_env()
    db_url = env.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Add your Neon connection string to the Modal secret "
            f"'{SECRET_NAME}' as DATABASE_URL (or NEON_DATABASE_URL)."
        )

    parsed = urlparse(db_url)
    query_raw = parse_qs(parsed.query)
    query = {
        key.lower(): [value.lower() for value in values if value]
        for key, values in query_raw.items()
    }

    if ENFORCE_DB_SSLMODE:
        sslmode_values = query.get("sslmode", [])
        if not any(mode in SECURE_SSLMODES for mode in sslmode_values):
            raise RuntimeError(
                "DATABASE_URL must include a secure sslmode in the query string: "
                f"{', '.join(sorted(SECURE_SSLMODES))}. "
                "Update your Modal secret "
                f"'{SECRET_NAME}' to include e.g. ?sslmode=require "
                "(or set MODAL_ENFORCE_DB_SSLMODE=0 to bypass)."
            )

    snippet = "from sqlalchemy import text; from app.db.engine import get_engine; e=get_engine(); c=e.connect(); c.execute(text('select 1')); c.close(); print('db ok')"
    _run_cmd(["-c", snippet])


def _verify_gpu_runtime() -> None:
    snippet = """
import subprocess
import sys
import torch

print(f"torch_version={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()}")
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
try:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout:
        print(probe.stdout.strip())
    if probe.stderr:
        print(probe.stderr.strip(), file=sys.stderr)
except Exception as exc:  # noqa: BLE001
    print(f"nvidia-smi probe skipped: {exc}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Modal GPU container.")
"""
    _run_cmd(["-c", snippet])


def _latest_calibration_path() -> str | None:
    candidates = sorted(
        REMOTE_CALIBRATION_DIR.glob("forecast_calibration_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    return str(candidates[-1])


def _run_collect_pipeline() -> None:
    _begin_run()
    _run_cmd(["-m", "alembic", "upgrade", "head"])
    _run_cmd(
        [
            "-m",
            "scripts.prizepicks.collect_prizepicks",
            "--output-dir",
            str(REMOTE_DATA_DIR / "snapshots"),
        ]
    )

    cmd = [
        "-m",
        "scripts.ml.run_top_picks_ensemble",
        "--models-dir",
        str(REMOTE_MODELS_DIR),
        "--ensemble-weights",
        str(REMOTE_MODELS_DIR / "ensemble_weights.json"),
        "--log-decisions",
        "--log-path",
        str(REMOTE_MONITORING_LOG),
    ]
    calibration = _latest_calibration_path()
    if calibration:
        cmd.extend(["--calibration", calibration])
    _run_cmd(cmd)
    _finish_run()


@app.function(
    schedule=modal.Cron("0 */3 * * *", timezone=SCHEDULE_TIMEZONE),
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def collect_every_3h() -> None:
    _run_collect_pipeline()


def _central_date_window() -> tuple[str, str]:
    today = datetime.now(ZoneInfo("America/Chicago")).date()
    end = today - timedelta(days=1)
    start = end - timedelta(days=2)
    return start.isoformat(), end.isoformat()


@app.function(
    gpu=TRAIN_GPU_SPEC,
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=15.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def train_nn_gpu() -> None:
    _begin_run()
    _verify_gpu_runtime()
    _run_cmd(
        [
            "-m",
            "scripts.ml.train_nn_model",
            "--model-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    _finish_run()


@app.function(
    gpu=TRAIN_GPU_SPEC,
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=15.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def train_tabdl_gpu() -> None:
    _begin_run()
    _verify_gpu_runtime()
    _run_cmd(
        [
            "-m",
            "scripts.ml.train_tabdl_model",
            "--model-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    _finish_run()


@app.function(
    gpu=TRAIN_GPU_SPEC,
    timeout=4 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=20.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def tune_tabdl_gpu(trials: int = 24) -> None:
    _begin_run()
    _verify_gpu_runtime()
    _run_cmd(
        [
            "-m",
            "scripts.ml.tune_tabdl_model",
            "--trials",
            str(max(4, int(trials))),
            "--output",
            str(REMOTE_DATA_DIR / "tuning" / "best_params_tabdl.json"),
            "--model-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    _finish_run()


@app.function(
    gpu=TRAIN_GPU_SPEC,
    timeout=15 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=5.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def gpu_smoke_test() -> None:
    _begin_run(verify_database=False)
    _verify_gpu_runtime()
    _finish_run()


def _run_train_pipeline() -> None:
    _begin_run()
    _run_cmd(["-m", "alembic", "upgrade", "head"])

    date_from, date_to = _central_date_window()
    _run_cmd(
        [
            "-m",
            "scripts.nba.fetch_nba_stats",
            "--date-from",
            date_from,
            "--date-to",
            date_to,
            "--allow-empty-on-failure",
            "--fast-fail-consecutive-timeouts",
            "2",
        ],
        env_overrides={
            "NBA_STATS_TIMEOUT_SECONDS": "15",
            "NBA_STATS_MAX_RETRIES": "1",
            "NBA_STATS_BACKOFF_SECONDS": "0.75",
        },
    )
    _run_cmd(
        [
            "-m",
            "scripts.ops.resolve_projection_outcomes",
            "--days-back",
            "30",
            "--decision-lag-hours",
            "3",
        ]
    )
    # Generate context-aware priors for shrinkage (uses resolved data)
    _run_cmd(
        [
            "-c",
            (
                "from app.ml.context_prior import compute_context_priors_from_db, save_context_priors; "
                "from app.db.engine import get_engine; "
                "from app.ml.artifact_store import upload_file; "
                "from pathlib import Path; "
                f"e = get_engine(); "
                f"save_context_priors(compute_context_priors_from_db(e, days_back=90), '{REMOTE_CONTEXT_PRIORS}'); "
                f"upload_file(e, model_name='context_priors', file_path=Path('{REMOTE_CONTEXT_PRIORS}'))"
            ),
        ],
        allow_fail=True,
    )
    freshness_rc = _run_cmd(
        [
            "-m",
            "scripts.ops.check_training_data_freshness",
            "--date-from",
            date_from,
            "--date-to",
            date_to,
            "--min-pending-games",
            "2",
            "--min-game-coverage-ratio",
            "0.60",
            "--min-stats-rows",
            "20",
            "--json-out",
            str(REMOTE_TRAIN_FRESHNESS_REPORT),
        ],
        allow_fail=True,
    )
    if freshness_rc != 0:
        print(
            "[modal] training deferred: NBA boxscore freshness gate failed "
            f"for window {date_from}..{date_to}. Keeping prior model artifacts."
        )
        _finish_run()
        return
    _run_cmd(
        [
            "-m",
            "scripts.ml.train_baseline_model",
            "--model-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    # Launch GPU training in parallel (non-blocking) while CPU models train
    nn_handle = train_nn_gpu.spawn()
    tabdl_handle = train_tabdl_gpu.spawn()
    print("[modal] NN and TabDL GPU training spawned in parallel")
    _run_cmd(
        [
            "-m",
            "scripts.ml.train_xgb_model",
            "--model-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    _run_cmd(
        [
            "-m",
            "scripts.ml.train_lgbm_model",
            "--models-dir",
            str(REMOTE_MODELS_DIR),
        ]
    )
    # Await GPU jobs before OOF generation (needs all expert models)
    print("[modal] Waiting for NN GPU training to complete...")
    try:
        nn_handle.get()
        print("[modal] NN GPU training completed successfully")
    except Exception as exc:
        print(f"[modal] WARNING: NN GPU training failed: {exc}")
    print("[modal] Waiting for TabDL GPU training to complete...")
    try:
        tabdl_handle.get()
        print("[modal] TabDL GPU training completed successfully")
    except Exception as exc:
        print(f"[modal] WARNING: TabDL GPU training failed: {exc}")
    oof_path = REMOTE_DATA_DIR / "oof_predictions.csv"
    _run_cmd(
        [
            "-m",
            "scripts.ml.generate_oof_predictions",
            "--output",
            str(oof_path),
        ],
        allow_fail=True,
    )
    _run_cmd(
        [
            "-m",
            "scripts.ops.monitor_model_health",
            "--ensemble-weights",
            str(REMOTE_MODELS_DIR / "ensemble_weights.json"),
            "--output",
            str(REMOTE_HEALTH_REPORT),
            "--alert-email",
            "--upload-db",
        ],
        allow_fail=True,
    )
    # Fit per-stat-type isotonic calibrators
    _run_cmd(
        [
            "-m",
            "scripts.ml.fit_stat_calibrator",
            "--days-back",
            "45",
            "--output",
            str(REMOTE_MODELS_DIR / "stat_calibrator.joblib"),
            "--upload-db",
        ],
        allow_fail=True,
    )
    # Ablation report — compare ensemble components
    _run_cmd(
        [
            "-m",
            "scripts.ops.ablation_report",
            "--output",
            str(REMOTE_ABLATION_REPORT),
        ],
        allow_fail=True,
    )
    # Drift detection — exit code 1 means drift detected
    drift_report = REMOTE_DATA_DIR / "reports" / "drift_report.json"
    drift_rc = _run_cmd(
        [
            "-m",
            "scripts.ops.check_drift",
            "--recent-days",
            "7",
            "--baseline-days",
            "30",
            "--output",
            str(drift_report),
        ],
        allow_fail=True,
    )
    if drift_rc != 0:
        print("[modal] Drift detected — triggering conditional hyperparameter retune")
        # CPU tuning for XGB/LGBM
        _run_cmd(
            [
                "-m",
                "scripts.ml.tune_hyperparams",
                "--model",
                "both",
                "--n-trials",
                "30",
                "--output-dir",
                str(REMOTE_DATA_DIR / "tuning"),
            ],
            allow_fail=True,
        )
        # GPU tuning for NN (spawn on GPU container)
        try:
            print("[modal] Spawning NN hyperparameter tuning on GPU...")
            tune_handle = tune_nn_on_drift.spawn()
            tune_handle.get()
            print("[modal] NN hyperparameter tuning completed")
        except Exception as exc:
            print(f"[modal] WARNING: NN hyperparameter tuning failed: {exc}")
    else:
        print("[modal] No drift detected — skipping hyperparameter retune")
    # Generate visual reports (advisory)
    _run_cmd(
        [
            "-m",
            "scripts.ml.generate_reports",
            "--output-dir",
            str(REMOTE_DATA_DIR / "reports"),
            "--weight-history",
            str(REMOTE_DATA_DIR / "reports" / "weight_history.jsonl"),
            "--drift-report",
            str(drift_report),
        ],
        allow_fail=True,
    )
    _finish_run()


@app.function(
    gpu=TRAIN_GPU_SPEC,
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=15.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def tune_nn_on_drift(n_trials: int = 15) -> None:
    """NN hyperparameter tuning triggered by drift detection (needs GPU)."""
    _begin_run()
    _verify_gpu_runtime()
    _run_cmd(
        [
            "-m",
            "scripts.ml.tune_nn_hyperparams",
            "--n-trials",
            str(max(5, int(n_trials))),
            "--output-dir",
            str(REMOTE_DATA_DIR / "tuning"),
        ],
        allow_fail=True,
    )
    _finish_run()


def _run_weekly_calibration() -> None:
    """Build backtest dataset and recalibrate the forecast distribution model."""
    _begin_run()
    _run_cmd(["-m", "alembic", "upgrade", "head"])

    backtest_csv = REMOTE_DATA_DIR / "calibration" / "forecast_backtest.csv"
    _run_cmd(
        [
            "-m",
            "scripts.calibration.build_forecast_backtest_dataset",
            "--output",
            str(backtest_csv),
            "--progress-every",
            "10000",
        ]
    )

    today = datetime.now(ZoneInfo("America/Chicago")).date()
    asof = today.isoformat()
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d_%H%M%S")
    calibration_json = REMOTE_CALIBRATION_DIR / f"forecast_calibration_{timestamp}.json"
    report_csv = REMOTE_DATA_DIR / "reports" / f"calibration_report_{timestamp}.csv"
    _run_cmd(
        [
            "-m",
            "scripts.calibration.run_weekly_calibration_report",
            "--dataset",
            str(backtest_csv),
            "--asof",
            asof,
            "--out-calibration",
            str(calibration_json),
            "--out-report",
            str(report_csv),
        ]
    )
    _finish_run()


@app.function(
    schedule=modal.Cron("0 14 * * *", timezone=SCHEDULE_TIMEZONE),
    timeout=8 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=30.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def train_daily() -> None:
    _run_train_pipeline()


@app.function(
    schedule=modal.Cron("0 10 * * 1", timezone=SCHEDULE_TIMEZONE),
    timeout=4 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=30.0, backoff_coefficient=2.0),
    max_containers=1,
    **shared_kwargs,
)
def calibrate_weekly() -> None:
    """Weekly recalibration of the forecast distribution model (Mondays 10am CT)."""
    _run_weekly_calibration()


@app.local_entrypoint()
def collect_now() -> None:
    collect_every_3h.remote()


@app.local_entrypoint()
def train_now() -> None:
    train_daily.remote()


@app.local_entrypoint()
def calibrate_now() -> None:
    calibrate_weekly.remote()


@app.local_entrypoint()
def gpu_check() -> None:
    gpu_smoke_test.remote()


@app.local_entrypoint()
def tune_tabdl_now(trials: int = 24) -> None:
    tune_tabdl_gpu.remote(trials=trials)
