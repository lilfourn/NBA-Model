from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline_model import load_env  # noqa: E402


def _norm_cdf_np(z: np.ndarray) -> np.ndarray:
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    zt = torch.from_numpy(z.astype(np.float32))
    u = 0.5 * (1.0 + torch.erf(zt / math.sqrt(2.0)))
    return u.numpy()


def _norm_pdf_np(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    sg = np.maximum(sigma, 1e-6)
    z = (y - mu) / sg
    return float(np.mean(np.log(sg) + 0.5 * z * z))


def _crps_normal(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """
    CRPS for Normal forecast (closed form).
    CRPS = sigma * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ]
    """
    sg = np.maximum(sigma, 1e-6)
    z = (y - mu) / sg
    Phi = _norm_cdf_np(z)
    phi = _norm_pdf_np(z)
    crps = sg * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def _pit(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    sg = np.maximum(sigma, 1e-6)
    z = (y - mu) / sg
    return np.clip(_norm_cdf_np(z), 0.0, 1.0)


def _ks_uniform(u: np.ndarray) -> float:
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = len(u)
    if n == 0:
        return 0.0
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - u)))


def _interval_coverage(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, *, level: float) -> float:
    # Central interval coverage for Normal. Hard-coded z to avoid scipy.
    z_map = {
        0.50: 0.67448975,
        0.80: 1.28155157,
        0.90: 1.64485363,
        0.95: 1.95996398,
    }
    z = z_map.get(level)
    if z is None:
        raise ValueError(f"Unsupported level {level}. Add z-value to z_map.")
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def _recency_weights(game_date: pd.Series, *, asof: pd.Timestamp, tau_days: float) -> np.ndarray:
    # w = exp(-Δdays/tau). Δdays<0 treated as 0.
    dt_days = (asof - game_date).dt.total_seconds().to_numpy(np.float64) / (24.0 * 3600.0)
    dt_days = np.maximum(dt_days, 0.0)
    return np.exp(-dt_days / tau_days).astype(np.float32)


def _time_windows(
    *,
    asof: pd.Timestamp,
    train_days: int,
    gap_days: int,
    val_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    val_end = asof
    val_start = asof - pd.Timedelta(days=val_days)
    train_end = val_start - pd.Timedelta(days=gap_days)
    train_start = train_end - pd.Timedelta(days=train_days)
    return train_start, train_end, val_start, val_end


def _build_pit_map(u: np.ndarray, *, knots: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.clip(u, 0.0, 1.0)
    u_sorted = np.sort(u)
    n = len(u_sorted)
    y = (np.arange(1, n + 1) - 0.5) / n
    if n <= knots:
        return u_sorted.astype(np.float32), y.astype(np.float32)
    idx = np.linspace(0, n - 1, knots).astype(int)
    return u_sorted[idx].astype(np.float32), y[idx].astype(np.float32)


def _fit_gaussian_recalibration_weighted(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    n_eff: np.ndarray | None,
    use_neff_inflation: bool,
    iters: int,
    lr: float,
    device: str,
) -> dict[str, Any]:
    """
    Fit by weighted Normal NLL:
      mu' = a*mu + b
      sigma_base = sqrt((c*sigma)^2 + d^2)
      sigma' = sqrt(sigma_base^2 + eta/(n_eff+1))   [optional]
    """
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sg_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    if n_eff is None:
        ne_t = torch.ones_like(mu_t)
    else:
        ne_t = torch.tensor(n_eff, dtype=torch.float32, device=device)

    a = torch.nn.Parameter(torch.tensor(1.0, device=device))
    b = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_c = torch.nn.Parameter(torch.tensor(0.0, device=device))  # c = softplus
    log_d = torch.nn.Parameter(torch.tensor(-2.0, device=device))  # d = softplus
    log_eta = torch.nn.Parameter(torch.tensor(-2.0, device=device))  # eta = softplus

    softplus = torch.nn.Softplus()
    opt = torch.optim.Adam([a, b, log_c, log_d, log_eta], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        c = softplus(log_c) + 1e-6
        d = softplus(log_d) + 1e-6
        eta = softplus(log_eta) + 1e-8

        mu_p = a * mu_t + b
        sg_base = torch.sqrt((c * sg_t) ** 2 + d**2)
        if use_neff_inflation:
            sg_p = torch.sqrt(sg_base**2 + eta / (ne_t + 1.0))
        else:
            sg_p = sg_base

        z = (y_t - mu_p) / sg_p
        nll = torch.log(sg_p) + 0.5 * z * z
        loss = torch.sum(w_t * nll) / (torch.sum(w_t) + 1e-8)
        loss.backward()
        opt.step()

    with torch.no_grad():
        c_val = float((softplus(log_c) + 1e-6).cpu().item())
        d_val = float((softplus(log_d) + 1e-6).cpu().item())
        eta_val = float((softplus(log_eta) + 1e-8).cpu().item())

    return {
        "a": float(a.detach().cpu().item()),
        "b": float(b.detach().cpu().item()),
        "c": c_val,
        "d": d_val,
        "eta": eta_val,
        "use_neff_inflation": bool(use_neff_inflation),
    }


def _apply_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    params: dict[str, Any],
    *,
    n_eff: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 1.0))
    d = float(params.get("d", 0.0))
    mu_p = a * mu + b
    sg_base = np.sqrt((c * sigma) ** 2 + d**2)

    if params.get("use_neff_inflation", False):
        eta = float(params.get("eta", 0.0))
        ne = np.ones_like(mu) if n_eff is None else np.maximum(n_eff, 0.0)
        sg_p = np.sqrt(sg_base**2 + eta / (ne + 1.0))
    else:
        sg_p = sg_base
    return mu_p.astype(np.float32), sg_p.astype(np.float32)


def _evaluate_distribution(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = {
        "nll": _normal_nll(mu, sigma, y),
        "crps": _crps_normal(mu, sigma, y),
        "pit_ks": _ks_uniform(_pit(mu, sigma, y)),
    }
    for level in (0.50, 0.80, 0.90):
        out[f"cov_{int(level*100)}"] = _interval_coverage(mu, sigma, y, level=level)
    return out


def run_weekly_calibration(
    *,
    dataset_path: str,
    out_calibration_json: str,
    out_report_csv: str,
    asof: str,
    train_days: int,
    gap_days: int,
    val_days: int,
    tau_days: float,
    min_rows: int,
    knots: int,
    use_neff_inflation: bool,
    device: str,
    iters: int,
    lr: float,
) -> None:
    df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)
    required = {"stat_type", "game_date", "y_true", "mu_hat", "sigma_hat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stat_type", "game_date", "y_true", "mu_hat", "sigma_hat"]
    )
    df = df[df["sigma_hat"] > 1e-6].copy()
    if "n_eff" not in df.columns:
        df["n_eff"] = 10.0

    asof_ts = pd.to_datetime(asof)
    train_start, train_end, val_start, val_end = _time_windows(
        asof=asof_ts, train_days=train_days, gap_days=gap_days, val_days=val_days
    )

    dfw = df[(df["game_date"] >= train_start) & (df["game_date"] < val_end)].copy()

    os.makedirs(os.path.dirname(out_calibration_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_report_csv), exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    calib_obj: dict[str, Any] = {
        "version": 1,
        "asof": str(asof_ts.date()),
        "train_window": {"start": str(train_start.date()), "end": str(train_end.date())},
        "val_window": {"start": str(val_start.date()), "end": str(val_end.date())},
        "by_stat_type": {},
    }

    for stat_type, g in dfw.groupby("stat_type"):
        g_train = g[(g["game_date"] >= train_start) & (g["game_date"] < train_end)]
        g_val = g[(g["game_date"] >= val_start) & (g["game_date"] < val_end)]

        if len(g_train) < min_rows or len(g_val) < max(2000, min_rows // 10):
            continue

        mu_tr = g_train["mu_hat"].to_numpy(np.float32)
        sg_tr = g_train["sigma_hat"].to_numpy(np.float32)
        y_tr = g_train["y_true"].to_numpy(np.float32)
        ne_tr = g_train["n_eff"].to_numpy(np.float32)

        mu_va = g_val["mu_hat"].to_numpy(np.float32)
        sg_va = g_val["sigma_hat"].to_numpy(np.float32)
        y_va = g_val["y_true"].to_numpy(np.float32)
        ne_va = g_val["n_eff"].to_numpy(np.float32)

        w_tr = _recency_weights(g_train["game_date"], asof=asof_ts, tau_days=tau_days)

        params = _fit_gaussian_recalibration_weighted(
            mu_tr,
            sg_tr,
            y_tr,
            w_tr,
            n_eff=ne_tr,
            use_neff_inflation=use_neff_inflation,
            iters=iters,
            lr=lr,
            device=device,
        )

        mu_tr_p, sg_tr_p = _apply_params(mu_tr, sg_tr, params, n_eff=ne_tr)
        u_tr = _pit(mu_tr_p, sg_tr_p, y_tr)
        pit_x, pit_y = _build_pit_map(u_tr, knots=knots)

        before = _evaluate_distribution(mu_va, sg_va, y_va)
        mu_va_p, sg_va_p = _apply_params(mu_va, sg_va, params, n_eff=ne_va)
        after_param = _evaluate_distribution(mu_va_p, sg_va_p, y_va)

        u_va = _pit(mu_va_p, sg_va_p, y_va)
        u_va_mapped = np.interp(u_va, pit_x, pit_y, left=0.0, right=1.0)
        pit_ks_full = _ks_uniform(u_va_mapped)

        report_rows.append(
            {
                "stat_type": stat_type,
                "train_rows": int(len(g_train)),
                "val_rows": int(len(g_val)),
                "nll_before": before["nll"],
                "nll_after_param": after_param["nll"],
                "crps_before": before["crps"],
                "crps_after_param": after_param["crps"],
                "cov90_before": before["cov_90"],
                "cov90_after_param": after_param["cov_90"],
                "pit_ks_before": before["pit_ks"],
                "pit_ks_after_param": after_param["pit_ks"],
                "pit_ks_after_full": float(pit_ks_full),
                "a": params["a"],
                "b": params["b"],
                "c": params["c"],
                "d": params["d"],
                "eta": params["eta"],
                "use_neff_inflation": bool(params.get("use_neff_inflation", False)),
            }
        )

        calib_obj["by_stat_type"][str(stat_type)] = {
            "params": params,
            "pit_map": {"x": pit_x.tolist(), "y": pit_y.tolist()},
        }

    # Global fallback calibrator trained across all stat types.
    g_train_all = dfw[(dfw["game_date"] >= train_start) & (dfw["game_date"] < train_end)]
    g_val_all = dfw[(dfw["game_date"] >= val_start) & (dfw["game_date"] < val_end)]
    if len(g_train_all) >= min_rows and len(g_val_all) >= max(2000, min_rows // 10):
        mu_tr = g_train_all["mu_hat"].to_numpy(np.float32)
        sg_tr = g_train_all["sigma_hat"].to_numpy(np.float32)
        y_tr = g_train_all["y_true"].to_numpy(np.float32)
        ne_tr = g_train_all["n_eff"].to_numpy(np.float32)

        mu_va = g_val_all["mu_hat"].to_numpy(np.float32)
        sg_va = g_val_all["sigma_hat"].to_numpy(np.float32)
        y_va = g_val_all["y_true"].to_numpy(np.float32)
        ne_va = g_val_all["n_eff"].to_numpy(np.float32)

        w_tr = _recency_weights(g_train_all["game_date"], asof=asof_ts, tau_days=tau_days)

        params = _fit_gaussian_recalibration_weighted(
            mu_tr,
            sg_tr,
            y_tr,
            w_tr,
            n_eff=ne_tr,
            use_neff_inflation=use_neff_inflation,
            iters=iters,
            lr=lr,
            device=device,
        )

        mu_tr_p, sg_tr_p = _apply_params(mu_tr, sg_tr, params, n_eff=ne_tr)
        u_tr = _pit(mu_tr_p, sg_tr_p, y_tr)
        pit_x, pit_y = _build_pit_map(u_tr, knots=knots)

        before = _evaluate_distribution(mu_va, sg_va, y_va)
        mu_va_p, sg_va_p = _apply_params(mu_va, sg_va, params, n_eff=ne_va)
        after_param = _evaluate_distribution(mu_va_p, sg_va_p, y_va)

        u_va = _pit(mu_va_p, sg_va_p, y_va)
        u_va_mapped = np.interp(u_va, pit_x, pit_y, left=0.0, right=1.0)
        pit_ks_full = _ks_uniform(u_va_mapped)

        report_rows.append(
            {
                "stat_type": "__global__",
                "train_rows": int(len(g_train_all)),
                "val_rows": int(len(g_val_all)),
                "nll_before": before["nll"],
                "nll_after_param": after_param["nll"],
                "crps_before": before["crps"],
                "crps_after_param": after_param["crps"],
                "cov90_before": before["cov_90"],
                "cov90_after_param": after_param["cov_90"],
                "pit_ks_before": before["pit_ks"],
                "pit_ks_after_param": after_param["pit_ks"],
                "pit_ks_after_full": float(pit_ks_full),
                "a": params["a"],
                "b": params["b"],
                "c": params["c"],
                "d": params["d"],
                "eta": params["eta"],
                "use_neff_inflation": bool(params.get("use_neff_inflation", False)),
            }
        )

        calib_obj["by_stat_type"]["__global__"] = {
            "params": params,
            "pit_map": {"x": pit_x.tolist(), "y": pit_y.tolist()},
        }

    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        report_df = report_df.sort_values(["nll_after_param", "pit_ks_after_full"])
    report_df.to_csv(out_report_csv, index=False)
    Path(out_calibration_json).write_text(json.dumps(calib_obj, indent=2), encoding="utf-8")

    print(f"Wrote report -> {out_report_csv}")
    print(f"Wrote calibration -> {out_calibration_json}")
    print(f"Calibrated stat_types: {len(calib_obj['by_stat_type'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly forecast-distribution calibration + report (time split).")
    ap.add_argument("--dataset", required=True, help="Backtest csv/parquet with game_date, stat_type, y_true, mu_hat, sigma_hat, (optional) n_eff.")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD (end of validation window)")
    ap.add_argument("--out-calibration", required=True, help="Output calibration JSON path.")
    ap.add_argument("--out-report", required=True, help="Output report CSV path.")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--gap-days", type=int, default=2)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--tau-days", type=float, default=60.0)
    ap.add_argument("--min-rows", type=int, default=20000)
    ap.add_argument("--knots", type=int, default=512)
    ap.add_argument("--no-neff-inflation", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    load_env()

    run_weekly_calibration(
        dataset_path=args.dataset,
        out_calibration_json=args.out_calibration,
        out_report_csv=args.out_report,
        asof=args.asof,
        train_days=args.train_days,
        gap_days=args.gap_days,
        val_days=args.val_days,
        tau_days=args.tau_days,
        min_rows=args.min_rows,
        knots=args.knots,
        use_neff_inflation=(not args.no_neff_inflation),
        device=args.device,
        iters=args.iters,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
