from __future__ import annotations

from app.services.scoring import _select_empty_publishable_fallback


def _item(
    *,
    projection_id: str,
    edge: float,
    p_pick: float,
    n_eff: float | None = 12.0,
    conformal_set_size: int | None = 1,
    is_prior_only: bool = False,
    forecast_edge_ok: bool = True,
) -> dict:
    return {
        "projection_id": projection_id,
        "edge": edge,
        "p_pick": p_pick,
        "n_eff": n_eff,
        "conformal_set_size": conformal_set_size,
        "is_prior_only": is_prior_only,
        "forecast_edge_ok": forecast_edge_ok,
    }


def test_soft_fallback_prefers_strict_soft_quality() -> None:
    scored = [
        _item(projection_id="A", edge=4.0, p_pick=0.59),
        _item(projection_id="B", edge=3.2, p_pick=0.58),
        _item(projection_id="C", edge=10.0, p_pick=0.70, is_prior_only=True),
    ]

    selected = _select_empty_publishable_fallback(scored, top=2)
    assert [item["projection_id"] for item in selected] == ["A", "B"]


def test_soft_fallback_relaxes_when_strict_pool_empty() -> None:
    scored = [
        _item(projection_id="A", edge=1.2, p_pick=0.56, conformal_set_size=2),
        _item(projection_id="B", edge=1.1, p_pick=0.55, conformal_set_size=2),
    ]

    selected = _select_empty_publishable_fallback(scored, top=5)
    assert [item["projection_id"] for item in selected] == ["A", "B"]


def test_soft_fallback_excludes_prior_only_when_possible() -> None:
    scored = [
        _item(projection_id="A", edge=12.0, p_pick=0.75, is_prior_only=True),
        _item(projection_id="B", edge=0.5, p_pick=0.51, is_prior_only=False),
    ]

    selected = _select_empty_publishable_fallback(scored, top=5)
    assert selected[0]["projection_id"] == "B"


def test_soft_fallback_returns_scored_if_all_are_prior_only() -> None:
    scored = [
        _item(projection_id="A", edge=12.0, p_pick=0.75, is_prior_only=True),
        _item(projection_id="B", edge=11.0, p_pick=0.73, is_prior_only=True),
    ]

    selected = _select_empty_publishable_fallback(scored, top=1)
    assert [item["projection_id"] for item in selected] == ["A"]
