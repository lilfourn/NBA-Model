import type { ScoredPick } from "@/lib/api";

function Bar({ label, value }: { label: string; value: number | null }) {
  if (value === null || value === undefined) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="w-20 text-muted-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">N/A</span>
      </div>
    );
  }
  const pct = Math.round(value * 100);
  const isOver = value >= 0.5;
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-muted-foreground">{label}</span>
      <div className="relative h-4 w-40 rounded bg-white/[0.06] overflow-hidden">
        <div
          className={`absolute left-0 top-0 h-full rounded ${isOver ? "bg-emerald-500/60" : "bg-red-500/50"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-xs w-12">{pct}%</span>
    </div>
  );
}

export function ExpertBreakdown({ pick }: { pick: ScoredPick }) {
  return (
    <div className="space-y-2 py-3 px-4">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-2">
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Expert P(Over)
          </p>
          <Bar label="Forecast" value={pick.p_forecast_cal} />
          <Bar label="Neural Net" value={pick.p_nn} />
          <Bar label="LogReg" value={pick.p_lr} />
          <Bar label="XGBoost" value={pick.p_xgb} />
          <Bar label="LightGBM" value={pick.p_lgbm} />
          <Bar label={pick.p_meta !== null ? "Meta ★" : "Meta"} value={pick.p_meta} />
        </div>
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Edge Breakdown
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <span className="text-muted-foreground">Edge Score</span>
            <span className="font-mono font-semibold">{pick.edge}</span>
            <span className="text-muted-foreground">Grade</span>
            <span className="font-mono font-semibold">{pick.grade}</span>
            <span className="text-muted-foreground">Ensemble P</span>
            <span className="font-mono">{(pick.prob_over * 100).toFixed(1)}%</span>
            <span className="text-muted-foreground">Conformal</span>
            <span className="font-mono">
              {pick.conformal_set_size === 1 ? "Confident" : pick.conformal_set_size === 2 ? "Ambiguous" : "—"}
            </span>
          </div>
        </div>
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Data Quality
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <span className="text-muted-foreground">μ̂ (mean)</span>
            <span className="font-mono">
              {pick.mu_hat !== null ? pick.mu_hat.toFixed(2) : "—"}
            </span>
            <span className="text-muted-foreground">σ̂ (std)</span>
            <span className="font-mono">
              {pick.sigma_hat !== null ? pick.sigma_hat.toFixed(2) : "—"}
            </span>
            <span className="text-muted-foreground">n_eff</span>
            <span className="font-mono">
              {pick.n_eff !== null ? pick.n_eff.toFixed(1) : "—"}
            </span>
            <span className="text-muted-foreground">Calibration</span>
            <span className="font-mono">{pick.calibration_status}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
