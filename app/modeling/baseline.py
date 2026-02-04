from __future__ import annotations

import statistics
from datetime import date
from typing import Iterable

from app.modeling.game_logs import index_game_logs_by_player
from app.modeling.name_utils import normalize_player_name
from app.modeling.probability import confidence_from_probability, probability_over
from app.modeling.stat_mappings import stat_value
from app.modeling.stabilization import STABILIZATION_GAMES
from app.modeling.types import PlayerGameLog, Prediction, Projection


class BaselinePredictor:
    def __init__(
        self,
        game_logs: Iterable[PlayerGameLog],
        *,
        min_games: int = 5,
        model_version: str = "baseline_v1",
    ) -> None:
        self._game_logs = list(game_logs)
        self._game_logs_by_player = index_game_logs_by_player(self._game_logs)
        self._min_games = min_games
        self._model_version = model_version
        self._league_means: dict[tuple[str, date | None], float] = {}

    def _league_mean(self, stat_type: str, cutoff: date | None) -> float | None:
        cache_key = (stat_type, cutoff)
        if cache_key in self._league_means:
            return self._league_means[cache_key]
        values: list[float] = []
        for log in self._game_logs:
            if cutoff and log.game_date:
                if log.game_date >= cutoff:
                    continue
            elif cutoff and log.game_date is None:
                # Skip undated rows when filtering to avoid leakage.
                continue
            value = stat_value(stat_type, log.stats)
            if value is not None:
                values.append(value)
        if not values:
            return None
        mean = statistics.mean(values)
        self._league_means[cache_key] = mean
        return mean

    def predict(self, projection: Projection) -> Prediction | None:
        if projection.is_combo:
            return None
        stat_type = projection.stat_type
        if not stat_type:
            return None

        player_key = normalize_player_name(projection.player_name)
        logs = self._game_logs_by_player.get(player_key, [])
        if not logs:
            return None

        cutoff = projection.start_time.date() if projection.start_time else None
        values: list[float] = []
        for log in logs:
            if cutoff and log.game_date and log.game_date >= cutoff:
                continue
            if cutoff and log.game_date is None:
                continue
            value = stat_value(stat_type, log.stats)
            if value is not None:
                values.append(value)

        if len(values) < self._min_games:
            return None

        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0

        stabilization = STABILIZATION_GAMES.get(stat_type, 10.0)
        league_mean = self._league_mean(stat_type, cutoff)
        if league_mean is None:
            league_mean = mean
        regressed_mean = (mean * len(values) + league_mean * stabilization) / (
            len(values) + stabilization
        )

        prob_over_normal = probability_over(projection.line_score, regressed_mean, std)
        # Empirical probability with a Beta(1,1) prior for stability.
        wins = sum(1 for value in values if value > projection.line_score)
        prob_over_empirical = (wins + 1.0) / (len(values) + 2.0)
        # Blend: trust empirical more as sample grows.
        weight = min(1.0, len(values) / max(1.0, stabilization))
        prob_over = prob_over_empirical * weight + prob_over_normal * (1.0 - weight)
        confidence = confidence_from_probability(prob_over)
        pick = "OVER" if prob_over >= 0.5 else "UNDER"

        return Prediction(
            projection=projection,
            pick=pick,
            prob_over=prob_over,
            confidence=confidence,
            mean=regressed_mean,
            std=std,
            model_version=self._model_version,
            details={
                "player_games": len(values),
                "raw_mean": mean,
                "stabilization_games": stabilization,
                "prob_over_empirical": prob_over_empirical,
                "prob_over_normal": prob_over_normal,
                "blend_weight": weight,
            },
        )
