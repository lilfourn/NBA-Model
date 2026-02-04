from datetime import date, datetime, timezone

from app.modeling.baseline import BaselinePredictor
from app.modeling.types import PlayerGameLog, Projection


def test_baseline_predictor_excludes_future_games():
    logs = [
        PlayerGameLog(
            player_id="1",
            player_name="Test Player",
            game_date=date(2026, 2, 1),
            stats={"PTS": 10},
        ),
        PlayerGameLog(
            player_id="1",
            player_name="Test Player",
            game_date=date(2026, 2, 5),
            stats={"PTS": 30},
        ),
    ]
    predictor = BaselinePredictor(logs, min_games=1)
    projection = Projection(
        projection_id="p1",
        player_id="1",
        player_name="Test Player",
        stat_type="Points",
        line_score=15.0,
        start_time=datetime(2026, 2, 3, tzinfo=timezone.utc),
        game_id=None,
        event_type="player",
        projection_type="Single Stat",
        trending_count=None,
        is_today=True,
        is_combo=False,
    )
    prediction = predictor.predict(projection)
    assert prediction is not None
    assert prediction.details["player_games"] == 1
    assert prediction.details["raw_mean"] == 10
