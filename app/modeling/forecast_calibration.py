from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _apply_params(
    mu: float, sigma: float, params: dict[str, float], *, n_eff: float | None = None
) -> tuple[float, float]:
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 1.0))
    d = float(params.get("d", 0.0))
    mu_p = a * mu + b
    sigma_base = math.sqrt((c * sigma) ** 2 + d**2)

    use_neff = bool(params.get("use_neff_inflation", False))
    eta = float(params.get("eta", 0.0))
    if use_neff and eta > 0 and n_eff is not None:
        sigma_p = math.sqrt(sigma_base**2 + eta / (max(float(n_eff), 0.0) + 1.0))
    else:
        sigma_p = sigma_base
    return mu_p, sigma_p


@dataclass(frozen=True)
class StatTypeCalibrator:
    params: dict[str, float]
    pit_x: np.ndarray
    pit_y: np.ndarray

    def calibrate_gaussian(
        self, mu: float, sigma: float, *, n_eff: float | None = None
    ) -> tuple[float, float]:
        return _apply_params(mu, sigma, self.params, n_eff=n_eff)

    def p_under(
        self,
        *,
        mu: float,
        sigma: float,
        line: float,
        continuity: float = 0.5,
        n_eff: float | None = None,
    ) -> float:
        mu_p, sigma_p = self.calibrate_gaussian(mu, sigma, n_eff=n_eff)
        if sigma_p <= 0:
            return 0.5
        z = ((line + continuity) - mu_p) / sigma_p
        u = _normal_cdf(z)
        # g(u) via monotone interpolation
        return float(np.interp(u, self.pit_x, self.pit_y, left=0.0, right=1.0))

    def p_over(
        self,
        *,
        mu: float,
        sigma: float,
        line: float,
        continuity: float = 0.5,
        n_eff: float | None = None,
    ) -> float:
        return 1.0 - self.p_under(
            mu=mu, sigma=sigma, line=line, continuity=continuity, n_eff=n_eff
        )


class ForecastDistributionCalibrator:
    def __init__(self, by_stat_type: dict[str, StatTypeCalibrator]) -> None:
        self._by_stat_type = by_stat_type

    def get(self, stat_type: str) -> StatTypeCalibrator | None:
        return self._by_stat_type.get(stat_type)

    def get_with_fallback(self, stat_type: str) -> tuple[StatTypeCalibrator | None, str]:
        """
        Returns (calibrator, status) where status is one of:
        - per_stat
        - fallback_global
        - raw (no calibrator available)
        """
        per = self._by_stat_type.get(stat_type)
        if per is not None:
            return per, "per_stat"
        global_cal = self._by_stat_type.get("__global__")
        if global_cal is not None:
            return global_cal, "fallback_global"
        return None, "raw"

    @classmethod
    def load(cls, path: str | Path) -> "ForecastDistributionCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        by_stat: dict[str, StatTypeCalibrator] = {}
        for stat_type, item in (payload.get("by_stat_type") or {}).items():
            params = item.get("params") or {}
            pit_map = item.get("pit_map") or {}
            pit_x = np.array(pit_map.get("x") or [], dtype=np.float32)
            pit_y = np.array(pit_map.get("y") or [], dtype=np.float32)
            if pit_x.size == 0 or pit_y.size == 0:
                continue
            by_stat[str(stat_type)] = StatTypeCalibrator(params=params, pit_x=pit_x, pit_y=pit_y)
        return cls(by_stat)

    def dump_info(self) -> dict[str, Any]:
        return {"stat_types": sorted(self._by_stat_type.keys())}
