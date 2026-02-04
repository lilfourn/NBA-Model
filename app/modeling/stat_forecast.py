from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from app.modeling.game_logs import index_game_logs_by_player
from app.modeling.name_utils import normalize_player_name
from app.modeling.probability import confidence_from_probability
from app.modeling.stat_mappings import stat_value
from app.modeling.stabilization import STABILIZATION_GAMES
from app.modeling.types import PlayerGameLog, Prediction, Projection
from app.modeling.forecast_calibration import ForecastDistributionCalibrator


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def minutes_value(stats: dict) -> float | None:
    for key in ("MIN", "minutes", "MINUTES"):
        value = _to_float(stats.get(key))
        if value is not None:
            return value
    return None


def _days_between(a: date, b: date) -> int:
    return (a - b).days


def _weight(delta_days: int, minutes: float, tau: float) -> float:
    minutes_factor = min(max(minutes / 24.0, 0.0), 1.0)
    return math.exp(-delta_days / tau) * minutes_factor


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / denom


def _weighted_var(values: list[float], weights: list[float], mean: float) -> float:
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / denom


def _effective_n(weights: list[float]) -> float:
    denom = sum(w * w for w in weights)
    if denom <= 0:
        return 0.0
    total = sum(weights)
    return (total * total) / denom


@dataclass
class ForecastParams:
    tau_short: float = 7.0
    tau_long: float = 21.0
    minutes_prior: float = 28.0
    k_minutes: float = 8.0
    alpha_rate: float = 0.5
    beta_minutes: float = 0.5
    beta_b2b: float = -1.5
    beta_rest: float = 0.5
    min_std_rate: float = 0.01
    min_std_minutes: float = 1.0
    overdispersion: dict[str, float] | None = None


class LeaguePriors:
    def __init__(
        self,
        logs: Sequence[PlayerGameLog],
        *,
        stat_types: Sequence[str],
        minutes_prior: float = 28.0,
    ) -> None:
        self._minutes_prior = minutes_prior
        dates = sorted({log.game_date for log in logs if log.game_date is not None})
        self._dates: list[date] = dates
        self._minutes_mean_after: dict[date, float] = {}
        self._rate_mean_after: dict[str, dict[date, float]] = {s: {} for s in stat_types}

        logs_sorted = sorted([log for log in logs if log.game_date is not None], key=lambda x: x.game_date)
        sum_minutes = 0.0
        count_minutes = 0
        sum_rate: dict[str, float] = {s: 0.0 for s in stat_types}
        count_rate: dict[str, int] = {s: 0 for s in stat_types}

        idx = 0
        for current in dates:
            while idx < len(logs_sorted) and logs_sorted[idx].game_date == current:
                log = logs_sorted[idx]
                minutes = minutes_value(log.stats)
                if minutes is not None:
                    sum_minutes += float(minutes)
                    count_minutes += 1
                for stat_type in stat_types:
                    stat_total = stat_value(stat_type, log.stats)
                    if minutes is None or minutes <= 0 or stat_total is None:
                        continue
                    sum_rate[stat_type] += float(stat_total) / float(minutes)
                    count_rate[stat_type] += 1
                idx += 1

            self._minutes_mean_after[current] = (
                (sum_minutes / count_minutes) if count_minutes else self._minutes_prior
            )
            for stat_type in stat_types:
                self._rate_mean_after[stat_type][current] = (
                    (sum_rate[stat_type] / count_rate[stat_type]) if count_rate[stat_type] else 0.0
                )

    def _prev_date(self, cutoff: date) -> date | None:
        idx = bisect_left(self._dates, cutoff) - 1
        if idx < 0:
            return None
        return self._dates[idx]

    def minutes_mean(self, cutoff: date | None) -> float:
        if cutoff is None:
            if not self._dates:
                return self._minutes_prior
            return self._minutes_mean_after[self._dates[-1]]
        prev = self._prev_date(cutoff)
        if prev is None:
            return self._minutes_prior
        return self._minutes_mean_after.get(prev, self._minutes_prior)

    def rate_mean(self, stat_type: str, cutoff: date | None) -> float:
        if not self._dates:
            return 0.0
        if cutoff is None:
            return self._rate_mean_after.get(stat_type, {}).get(self._dates[-1], 0.0)
        prev = self._prev_date(cutoff)
        if prev is None:
            return 0.0
        return self._rate_mean_after.get(stat_type, {}).get(prev, 0.0)


class StatForecastPredictor:
    def __init__(
        self,
        game_logs: Iterable[PlayerGameLog],
        *,
        min_games: int = 5,
        params: ForecastParams | None = None,
        league_priors: LeaguePriors | None = None,
        calibrator: ForecastDistributionCalibrator | None = None,
    ) -> None:
        self._game_logs = list(game_logs)
        self._game_logs_by_player = index_game_logs_by_player(self._game_logs)
        self._min_games = min_games
        self._params = params or ForecastParams()
        self._league_priors = league_priors
        self._calibrator = calibrator
        self._league_rate_cache: dict[tuple[str, date | None], float] = {}
        self._league_minutes_cache: dict[date | None, float] = {}

    def _league_rate(self, stat_type: str, cutoff: date | None) -> float:
        if self._league_priors is not None:
            return self._league_priors.rate_mean(stat_type, cutoff)
        cache_key = (stat_type, cutoff)
        if cache_key in self._league_rate_cache:
            return self._league_rate_cache[cache_key]
        values: list[float] = []
        for log in self._game_logs:
            if cutoff and log.game_date and log.game_date >= cutoff:
                continue
            minutes = minutes_value(log.stats)
            stat_total = stat_value(stat_type, log.stats)
            if minutes is None or minutes <= 0 or stat_total is None:
                continue
            values.append(stat_total / minutes)
        mean = sum(values) / len(values) if values else 0.0
        self._league_rate_cache[cache_key] = mean
        return mean

    def _league_minutes(self, cutoff: date | None) -> float:
        if self._league_priors is not None:
            return self._league_priors.minutes_mean(cutoff)
        if cutoff in self._league_minutes_cache:
            return self._league_minutes_cache[cutoff]
        values: list[float] = []
        for log in self._game_logs:
            if cutoff and log.game_date and log.game_date >= cutoff:
                continue
            minutes = minutes_value(log.stats)
            if minutes is None:
                continue
            values.append(minutes)
        mean = sum(values) / len(values) if values else self._params.minutes_prior
        self._league_minutes_cache[cutoff] = mean
        return mean

    def _decision_date(self, projection: Projection) -> date | None:
        if projection.start_time:
            return projection.start_time.date()
        return None

    def _history(self, projection: Projection) -> list[PlayerGameLog]:
        key = normalize_player_name(projection.player_name)
        logs = self._game_logs_by_player.get(key, [])
        cutoff = self._decision_date(projection)
        if cutoff:
            return [log for log in logs if log.game_date and log.game_date < cutoff]
        return logs

    def _rest_days(self, cutoff: date | None, logs: list[PlayerGameLog]) -> tuple[float, float]:
        if not cutoff or not logs:
            return 0.0, 0.0
        last_game = logs[-1].game_date
        if not last_game:
            return 0.0, 0.0
        diff = (cutoff - last_game).days
        rest_days = max(diff - 1, 0)
        is_b2b = 1.0 if rest_days == 0 else 0.0
        return float(rest_days), float(is_b2b)

    def _forecast_mean_std(self, projection: Projection) -> tuple[float, float, dict[str, float]] | None:
        stat_type = projection.stat_type
        if not stat_type:
            return None

        cutoff = self._decision_date(projection)
        logs = self._history(projection)
        if len(logs) < self._min_games:
            return None

        params = self._params
        rate_values: list[float] = []
        minutes_values: list[float] = []
        days_values: list[int] = []

        for log in logs:
            if log.game_date is None or cutoff is None:
                continue
            minutes = minutes_value(log.stats)
            stat_total = stat_value(stat_type, log.stats)
            if minutes is None or minutes <= 0 or stat_total is None:
                continue
            rate_values.append(stat_total / minutes)
            minutes_values.append(minutes)
            days_values.append(_days_between(cutoff, log.game_date))

        if len(rate_values) < self._min_games:
            return None

        weights_short = [
            _weight(days, minutes, params.tau_short)
            for days, minutes in zip(days_values, minutes_values)
        ]
        weights_long = [
            _weight(days, minutes, params.tau_long)
            for days, minutes in zip(days_values, minutes_values)
        ]

        r_short = _weighted_mean(rate_values, weights_short)
        r_long = _weighted_mean(rate_values, weights_long)
        m_short = _weighted_mean(minutes_values, weights_short)
        m_long = _weighted_mean(minutes_values, weights_long)
        r_var = _weighted_var(rate_values, weights_long, r_long)
        m_var = _weighted_var(minutes_values, weights_long, m_long)
        n_eff = _effective_n(weights_long)

        league_rate = self._league_rate(stat_type, cutoff)
        league_minutes = self._league_minutes(cutoff)
        k_rate = float(STABILIZATION_GAMES.get(stat_type, 10.0))

        r_hat0 = (n_eff * r_long + k_rate * league_rate) / max(n_eff + k_rate, 1e-6)
        m_hat0 = (n_eff * m_long + params.k_minutes * league_minutes) / max(
            n_eff + params.k_minutes, 1e-6
        )

        delta_r = r_short - r_long
        delta_m = m_short - m_long
        rest_days, is_b2b = self._rest_days(cutoff, logs)
        rest_adj = min(rest_days, 3.0)

        r_hat = max(0.0, r_hat0 + params.alpha_rate * delta_r)
        m_hat = max(
            0.0,
            m_hat0 + params.beta_minutes * delta_m + params.beta_b2b * is_b2b + params.beta_rest * rest_adj,
        )

        mean = r_hat * m_hat

        var_r = r_var + params.min_std_rate ** 2
        var_m = m_var + params.min_std_minutes ** 2
        var = (m_hat * m_hat * var_r) + (r_hat * r_hat * var_m)

        overdisp = 0.0
        if params.overdispersion and stat_type in params.overdispersion:
            overdisp = params.overdispersion[stat_type] * mean
        var += overdisp

        std = math.sqrt(max(var, 1e-6))
        return mean, std, {
            "rate_short": r_short,
            "rate_long": r_long,
            "minutes_short": m_short,
            "minutes_long": m_long,
            "n_eff": n_eff,
            "rest_days": rest_days,
            "is_back_to_back": is_b2b,
            "league_rate": league_rate,
            "league_minutes": league_minutes,
        }

    def forecast_distribution(
        self, projection: Projection
    ) -> tuple[float, float, dict[str, float]] | None:
        """Return (mu_hat, sigma_hat, details) without applying any calibration."""
        return self._forecast_mean_std(projection)

    def predict(self, projection: Projection) -> Prediction | None:
        stat_type = projection.stat_type
        forecast = self._forecast_mean_std(projection)
        if forecast is None:
            return None
        mean_raw, std_raw, details = forecast

        line = projection.line_score
        if std_raw <= 0:
            return None

        # Always compute the raw normal-based probability for monitoring/debugging.
        z_raw = ((line + 0.5) - mean_raw) / std_raw
        prob_over_raw = 1.0 - 0.5 * (1.0 + math.erf(z_raw / math.sqrt(2.0)))

        mean = mean_raw
        std = std_raw
        prob_over = prob_over_raw
        calibration_status = "raw"
        if self._calibrator is not None:
            cal, status = self._calibrator.get_with_fallback(stat_type)
            if cal is not None:
                n_eff = details.get("n_eff")
                mean, std = cal.calibrate_gaussian(mean_raw, std_raw, n_eff=n_eff)
                prob_over = cal.p_over(mu=mean_raw, sigma=std_raw, line=line, n_eff=n_eff)
                calibration_status = status
            else:
                # Safe fallback to reduce extreme probabilities when uncalibrated.
                lam = 0.5
                prob_over = 0.5 + lam * (prob_over_raw - 0.5)
                calibration_status = "raw"
        else:
            lam = 0.5
            prob_over = 0.5 + lam * (prob_over_raw - 0.5)
            calibration_status = "raw"

        confidence = confidence_from_probability(prob_over)
        pick = "OVER" if prob_over >= 0.5 else "UNDER"

        return Prediction(
            projection=projection,
            pick=pick,
            prob_over=prob_over,
            confidence=confidence,
            mean=mean,
            std=std,
            model_version="stat_forecast_v1",
            details={
                **details,
                "raw_mean": mean_raw,
                "raw_std": std_raw,
                "prob_over_raw": prob_over_raw,
                "calibration_status": calibration_status,
            },
        )
