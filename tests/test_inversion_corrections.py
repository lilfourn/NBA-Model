from __future__ import annotations

import pandas as pd

import app.ml.inversion_corrections as inv


def test_load_inversion_flags_requires_min_sample_and_gain(monkeypatch) -> None:
    report = {
        "expert_metrics": {
            "p_forecast_cal": {
                "n": 50,
                "inversion_test": {
                    "accuracy": 0.45,
                    "accuracy_inverted": 0.55,
                    "logloss": 0.80,
                    "logloss_inverted": 0.72,
                },
            },
            "p_lr": {
                "n": 500,
                "inversion_test": {
                    "accuracy": 0.46,
                    "accuracy_inverted": 0.53,
                    "logloss": 0.79,
                    "logloss_inverted": 0.73,
                },
            },
        }
    }
    monkeypatch.setattr(inv, "_load_report", lambda *_args, **_kwargs: report)

    flags = inv.load_inversion_flags(
        object(),
        min_n=300,
        min_accuracy_gain=0.03,
        min_logloss_gain=0.02,
    )

    assert "p_forecast_cal" not in flags  # blocked by low n
    assert flags.get("p_lr") is True


def test_load_forecast_stat_inversion_flags_from_db(monkeypatch) -> None:
    rows: list[dict[str, object]] = []

    # Inverted stat: p_forecast_cal points the wrong direction.
    for idx in range(280):
        rows.append(
            {
                "stat_type": "Free Throws Made",
                "over_label": 0 if idx < 196 else 1,
                "p_forecast_cal": 0.8,
            }
        )

    # Healthy stat: original orientation is better.
    for idx in range(280):
        rows.append(
            {
                "stat_type": "Points",
                "over_label": 1 if idx < 196 else 0,
                "p_forecast_cal": 0.8,
            }
        )

    df = pd.DataFrame(rows)
    monkeypatch.setattr(inv.pd, "read_sql", lambda *_args, **_kwargs: df.copy())

    class _DummyEngine:
        def connect(self) -> None:
            return None

    flags = inv.load_forecast_stat_inversion_flags(
        _DummyEngine(),
        days_back=180,
        min_samples=200,
        min_accuracy_gain=0.05,
        min_logloss_gain=0.02,
    )

    assert flags.get("Free Throws Made") is True
    assert "Points" not in flags

