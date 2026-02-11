import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def normal_cdf_torch(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def fit_gaussian_recalibration(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    iters: int = 300,
    lr: float = 0.05,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Fit:
      mu' = a*mu + b
      sigma' = sqrt((c*sigma)^2 + d^2)
    by minimizing Normal negative log-likelihood.
    """
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sg_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    a = torch.nn.Parameter(torch.tensor(1.0, device=device))
    b = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_c = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_d = torch.nn.Parameter(torch.tensor(-2.0, device=device))

    softplus = torch.nn.Softplus()
    opt = torch.optim.Adam([a, b, log_c, log_d], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        c = softplus(log_c) + 1e-6
        d = softplus(log_d) + 1e-6

        mu_p = a * mu_t + b
        sg_p = torch.sqrt((c * sg_t) ** 2 + d**2)

        nll = torch.mean(torch.log(sg_p) + 0.5 * ((y_t - mu_p) / sg_p) ** 2)
        nll.backward()
        opt.step()

    with torch.no_grad():
        c_val = float((softplus(log_c) + 1e-6).cpu().item())
        d_val = float((softplus(log_d) + 1e-6).cpu().item())

    return {
        "a": float(a.detach().cpu().item()),
        "b": float(b.detach().cpu().item()),
        "c": c_val,
        "d": d_val,
    }


def apply_recalibration_params(
    mu: np.ndarray, sigma: np.ndarray, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])
    mu_p = a * mu + b
    sg_p = np.sqrt((c * sigma) ** 2 + d**2)
    return mu_p, sg_p


def build_empirical_cdf_map(u: np.ndarray, *, knots: int = 512) -> tuple[np.ndarray, np.ndarray]:
    u = np.clip(u, 0.0, 1.0)
    u_sorted = np.sort(u)
    n = len(u_sorted)
    cdf_vals = (np.arange(1, n + 1) - 0.5) / n
    if n <= knots:
        return u_sorted.astype(np.float32), cdf_vals.astype(np.float32)
    idx = np.linspace(0, n - 1, knots).astype(int)
    return u_sorted[idx].astype(np.float32), cdf_vals[idx].astype(np.float32)


def ks_to_uniform(u: np.ndarray) -> float:
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = len(u)
    if n == 0:
        return 0.0
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - u)))


def calibrate(
    df: pd.DataFrame,
    *,
    min_rows_per_stat: int = 5000,
    knots: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    required = {"stat_type", "y_true", "mu_hat", "sigma_hat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stat_type", "y_true", "mu_hat", "sigma_hat"]
    )
    df = df[df["sigma_hat"] > 1e-6].copy()

    out: dict[str, Any] = {"version": 1, "by_stat_type": {}}

    for stat_type, g in df.groupby("stat_type"):
        if len(g) < min_rows_per_stat:
            continue

        y = g["y_true"].to_numpy(np.float32)
        mu = g["mu_hat"].to_numpy(np.float32)
        sg = g["sigma_hat"].to_numpy(np.float32)

        params = fit_gaussian_recalibration(mu, sg, y, device=device)
        mu_p, sg_p = apply_recalibration_params(mu, sg, params)

        z = (y - mu_p) / (sg_p + 1e-6)
        # PIT values under recalibrated gaussian
        z_t = torch.tensor(z, dtype=torch.float32, device=device)
        u = normal_cdf_torch(z_t).detach().cpu().numpy()

        xk, yk = build_empirical_cdf_map(u, knots=knots)

        z0 = (y - mu) / (sg + 1e-6)
        z0_t = torch.tensor(z0, dtype=torch.float32, device=device)
        u0 = normal_cdf_torch(z0_t).detach().cpu().numpy()
        ks_before = ks_to_uniform(u0)
        ks_after = ks_to_uniform(u)

        out["by_stat_type"][str(stat_type)] = {
            "params": params,
            "pit_map": {"x": xk.tolist(), "y": yk.tolist()},
            "diagnostics": {"rows": int(len(g)), "ks_before": ks_before, "ks_after": ks_after},
        }

    # Fallback calibrator trained across all stat_types.
    if len(df) >= min_rows_per_stat:
        y = df["y_true"].to_numpy(np.float32)
        mu = df["mu_hat"].to_numpy(np.float32)
        sg = df["sigma_hat"].to_numpy(np.float32)

        params = fit_gaussian_recalibration(mu, sg, y, device=device)
        mu_p, sg_p = apply_recalibration_params(mu, sg, params)

        z = (y - mu_p) / (sg_p + 1e-6)
        z_t = torch.tensor(z, dtype=torch.float32, device=device)
        u = normal_cdf_torch(z_t).detach().cpu().numpy()

        xk, yk = build_empirical_cdf_map(u, knots=knots)

        z0 = (y - mu) / (sg + 1e-6)
        z0_t = torch.tensor(z0, dtype=torch.float32, device=device)
        u0 = normal_cdf_torch(z0_t).detach().cpu().numpy()
        ks_before = ks_to_uniform(u0)
        ks_after = ks_to_uniform(u)

        out["by_stat_type"]["__global__"] = {
            "params": params,
            "pit_map": {"x": xk.tolist(), "y": yk.tolist()},
            "diagnostics": {"rows": int(len(df)), "ks_before": ks_before, "ks_after": ks_after},
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate forecast distribution via MLE + PIT mapping.")
    ap.add_argument("--input", required=True, help="CSV with stat_type,y_true,mu_hat,sigma_hat")
    ap.add_argument("--output", required=True, help="Output calibration JSON path")
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--knots", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    payload = calibrate(
        df,
        min_rows_per_stat=args.min_rows,
        knots=args.knots,
        device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote calibration -> {output_path}")
    print(f"Calibrated stat_types: {len(payload['by_stat_type'])}")


if __name__ == "__main__":
    main()
