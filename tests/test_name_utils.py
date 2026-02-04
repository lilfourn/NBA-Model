import json
from pathlib import Path

from app.modeling.name_utils import normalize_player_name


def test_normalize_player_name_strips_suffixes():
    assert normalize_player_name("LeBron James Jr.") == "lebron james"


def test_normalize_player_name_overrides(tmp_path: Path, monkeypatch):
    overrides = {"kristaps porzingis": "kristaps porzingis"}
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps(overrides), encoding="utf-8")

    monkeypatch.setenv("PLAYER_NAME_OVERRIDES_PATH", str(overrides_path))
    from importlib import reload

    import app.core.config as config
    import app.modeling.name_utils as name_utils

    reload(config)
    reload(name_utils)
    assert name_utils.normalize_player_name("Kristaps Porziņģis") == "kristaps porzingis"
