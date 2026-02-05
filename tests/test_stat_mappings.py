from app.modeling.stat_mappings import stat_value


def test_stat_value_single_stats():
    stats = {
        "PTS": 25,
        "REB": 10,
        "AST": 5,
        "FG3M": 4,
        "FG3A": 9,
        "FGM": 9,
        "FGA": 18,
        "FTM": 3,
        "FTA": 4,
        "STL": 2,
        "BLK": 1,
        "TOV": 3,
        "PF": 2,
        "OREB": 3,
        "DREB": 7,
        "DUNKS": 2,
    }
    assert stat_value("Points", stats) == 25
    assert stat_value("Rebounds", stats) == 10
    assert stat_value("Assists", stats) == 5
    assert stat_value("3-PT Made", stats) == 4
    assert stat_value("FG Attempted", stats) == 18
    assert stat_value("Free Throws Made", stats) == 3
    assert stat_value("Steals", stats) == 2
    assert stat_value("Blocked Shots", stats) == 1
    assert stat_value("Dunks", stats) == 2
    assert stat_value("Turnovers", stats) == 3
    assert stat_value("Personal Fouls", stats) == 2
    assert stat_value("Offensive Rebounds", stats) == 3
    assert stat_value("Defensive Rebounds", stats) == 7


def test_stat_value_combo_stats():
    stats = {"PTS": 20, "REB": 8, "AST": 6, "BLK": 2, "STL": 1}
    assert stat_value("Pts+Rebs", stats) == 28
    assert stat_value("Pts+Asts", stats) == 26
    assert stat_value("Rebs+Asts", stats) == 14
    assert stat_value("Pts+Rebs+Asts", stats) == 34
    assert stat_value("Blks+Stls", stats) == 3


def test_stat_value_two_pointers():
    stats = {"FGM": 9, "FG3M": 4, "FGA": 18, "FG3A": 8}
    assert stat_value("Two Pointers Made", stats) == 5
    assert stat_value("Two Pointers Attempted", stats) == 10


def test_stat_value_unknown():
    assert stat_value("Quarters with 3+ Points", {"PTS": 10}) is None


def test_stat_value_fantasy_score():
    stats = {"PTS": 20, "REB": 10, "AST": 5, "STL": 2, "BLK": 1, "TOV": 3}
    # 20 + 1.2*10 + 1.5*5 + 3*2 + 3*1 - 3 = 45.5
    assert stat_value("Fantasy Score", stats) == 45.5


def test_stat_value_returns_none_on_nonfinite_inputs():
    assert stat_value("Points", {"PTS": float("nan")}) is None
    assert stat_value("Points", {"PTS": "nan"}) is None
    assert stat_value("Pts+Rebs", {"PTS": 10, "REB": float("inf")}) is None
