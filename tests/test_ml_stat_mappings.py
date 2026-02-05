import pandas as pd

from app.ml.feature_engineering import compute_league_means
from app.ml.stat_mappings import stat_value_from_row


def test_stat_value_from_row_dunks() -> None:
    assert stat_value_from_row("Dunks", {"dunks": 3}) == 3.0


def test_stat_value_from_row_fantasy_score() -> None:
    row = {"points": 20, "rebounds": 10, "assists": 5, "steals": 2, "blocks": 1, "turnovers": 3}
    assert stat_value_from_row("Fantasy Score", row) == 45.5


def test_compute_league_means_includes_fantasy_score_and_dunks() -> None:
    df = pd.DataFrame(
        [
            {"points": 10, "rebounds": 5, "assists": 4, "steals": 1, "blocks": 0, "turnovers": 2, "dunks": 1},
            {"points": 20, "rebounds": 10, "assists": 5, "steals": 2, "blocks": 1, "turnovers": 3, "dunks": 2},
        ]
    )
    means = compute_league_means(df)
    # fantasyscore key is normalized form
    assert "fantasyscore" in means
    assert "dunks" in means

