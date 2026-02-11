// Expert/model display metadata shared across charts
export const EXPERT_META: Record<string, { label: string; color: string }> = {
  p_forecast_cal: { label: "Forecast", color: "oklch(0.75 0.12 75)" },
  p_nn: { label: "NN", color: "oklch(0.6 0.04 250)" },
  p_tabdl: { label: "TabDL", color: "oklch(0.62 0.08 30)" },
  p_lr: { label: "LR", color: "oklch(0.55 0.03 180)" },
  p_xgb: { label: "XGB", color: "oklch(0.65 0.06 60)" },
  p_lgbm: { label: "LGBM", color: "oklch(0.5 0 0)" },
};

export const MODEL_META: Record<string, { label: string; color: string }> = {
  baseline_logreg: { label: "LR", color: "oklch(0.75 0.12 75)" },
  nn_gru_attention: { label: "NN", color: "oklch(0.6 0.04 250)" },
  tabdl_mlp: { label: "TabDL", color: "oklch(0.62 0.08 30)" },
  xgboost: { label: "XGB", color: "oklch(0.55 0.03 180)" },
  xgb: { label: "XGB", color: "oklch(0.55 0.03 180)" },
  lightgbm: { label: "LGBM", color: "oklch(0.65 0.06 60)" },
  lgbm: { label: "LGBM", color: "oklch(0.65 0.06 60)" },
  meta_learner: { label: "Meta", color: "oklch(0.5 0 0)" },
};

export const EXPERT_COLORS: Record<string, string> = {
  p_forecast_cal: "oklch(0.75 0.12 75)",
  p_nn: "oklch(0.6 0.04 250)",
  p_tabdl: "oklch(0.62 0.08 30)",
  p_lr: "oklch(0.55 0.03 180)",
  p_xgb: "oklch(0.65 0.06 60)",
  p_lgbm: "oklch(0.5 0 0)",
};

export const EXPERT_LABELS: Record<string, string> = {
  p_forecast_cal: "Forecast",
  p_nn: "NN",
  p_tabdl: "TabDL",
  p_lr: "LR",
  p_xgb: "XGB",
  p_lgbm: "LGBM",
};

// Stat type abbreviations for compact display
export const STAT_ABBREV: Record<string, string> = {
  "3-PT Made": "3PM",
  "Assists": "AST",
  "Blks+Stls": "BLK+STL",
  "Blocked Shots": "BLK",
  "FG Attempted": "FGA",
  "FG Made": "FGM",
  "Fantasy Score": "FPTS",
  "Free Throws Attempted": "FTA",
  "Free Throws Made": "FTM",
  "Points": "PTS",
  "Pts+Asts": "PTS+AST",
  "Pts+Rebs+Asts": "PRA",
  "Rebounds": "REB",
  "Rebs+Asts": "REB+AST",
  "Steals": "STL",
  "Turnovers": "TO",
  "Two Pointers Attempted": "2PA",
  "Two Pointers Made": "2PM",
};

export function abbrevStat(s: string): string {
  return STAT_ABBREV[s] ?? s;
}

// Common chart margins and settings
export const CHART_MARGINS = { top: 5, right: 12, bottom: 5, left: 0 };
export const CHART_GRID = { vertical: false, strokeOpacity: 0.06 as const };
