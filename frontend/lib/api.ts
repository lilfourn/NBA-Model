const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ScoredPick {
  projection_id: string;
  player_name: string;
  player_image_url: string | null;
  player_id: string;
  game_id: string | null;
  stat_type: string;
  line_score: number;
  pick: "OVER" | "UNDER";
  prob_over: number;
  confidence: number;
  rank_score: number;
  p_forecast_cal: number | null;
  p_nn: number | null;
  p_tabdl: number | null;
  p_lr: number | null;
  p_xgb: number | null;
  p_lgbm: number | null;
  p_meta: number | null;
  mu_hat: number | null;
  sigma_hat: number | null;
  calibration_status: string;
  n_eff: number | null;
  conformal_set_size: number | null;
  edge: number;
  grade: string;
}

export interface ScoringResult {
  snapshot_id: string;
  scored_at: string;
  total_scored: number;
  picks: ScoredPick[];
}

export interface Snapshot {
  id: string;
  fetched_at: string | null;
  data_count: number | null;
  included_count: number | null;
}

export interface SnapshotsResponse {
  count: number;
  snapshots: Snapshot[];
}

const PICKS_CACHE_KEY = "nba_picks_cache";

export function getCachedPicks(): ScoringResult | null {
  try {
    const raw = localStorage.getItem(PICKS_CACHE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as ScoringResult;
  } catch {
    return null;
  }
}

export function setCachedPicks(result: ScoringResult): void {
  try {
    localStorage.setItem(PICKS_CACHE_KEY, JSON.stringify(result));
  } catch {
    // Storage full or unavailable — ignore
  }
}

export async function fetchPicks(params?: {
  snapshot_id?: string;
  game_date?: string;
  top?: number;
  rank?: string;
  include_non_today?: boolean;
  force?: boolean;
}): Promise<ScoringResult> {
  const searchParams = new URLSearchParams();
  if (params?.snapshot_id) searchParams.set("snapshot_id", params.snapshot_id);
  if (params?.game_date) searchParams.set("game_date", params.game_date);
  if (params?.top) searchParams.set("top", params.top.toString());
  if (params?.rank) searchParams.set("rank", params.rank);
  if (params?.include_non_today)
    searchParams.set("include_non_today", "true");
  if (params?.force) searchParams.set("force", "true");

  const url = `${API_URL}/api/picks?${searchParams.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const result: ScoringResult = await res.json();
  setCachedPicks(result);
  return result;
}

export async function fetchSnapshots(
  limit: number = 20
): Promise<SnapshotsResponse> {
  const res = await fetch(`${API_URL}/api/snapshots-list?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// --- Jobs / Operations ---

export interface Job {
  id: string;
  job_type: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  output: string;
  error: string | null;
}

export interface JobType {
  value: string;
  label: string;
}

export async function fetchJobTypes(): Promise<JobType[]> {
  const res = await fetch(`${API_URL}/api/job-types`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.job_types;
}

export async function startJob(jobType: string): Promise<Job> {
  const res = await fetch(`${API_URL}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type: jobType }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export async function fetchJobs(limit: number = 20): Promise<Job[]> {
  const res = await fetch(`${API_URL}/api/jobs?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.jobs;
}

export async function fetchJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_URL}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// --- Stats / Model Metrics ---

export interface TrainingRun {
  id: string;
  created_at: string | null;
  model_name: string;
  train_rows: number;
  accuracy: number | null;
  roc_auc: number | null;
  logloss: number | null;
  conformal_q_hat: number | null;
}

export interface TrainingHistoryResponse {
  runs: TrainingRun[];
}

export interface ExpertSummary {
  model_name: string;
  accuracy: number | null;
  roc_auc: number | null;
  logloss: number | null;
  conformal_q_hat: number | null;
  train_rows: number;
  created_at: string | null;
}

export interface ExpertComparisonResponse {
  experts: ExpertSummary[];
}

export interface HitRatePoint {
  index: number;
  ensemble_hit_rate: number | null;
  p_forecast_cal_hit_rate: number | null;
  p_nn_hit_rate: number | null;
  p_lr_hit_rate: number | null;
  p_xgb_hit_rate: number | null;
  p_lgbm_hit_rate: number | null;
  date: string | null;
}

export interface HitRateResponse {
  total_predictions: number;
  total_resolved: number;
  total_scored: number;
  overall_hit_rate: number | null;
  published_hit_rate: number | null;
  published_n: number;
  placed_hit_rate: number | null;
  placed_n: number;
  coverage: number;
  actionable_threshold: number;
  rolling: HitRatePoint[];
}

export interface CalibrationEntry {
  stat_type: string;
  train_rows: number | null;
  val_rows: number | null;
  nll_before: number | null;
  nll_after: number | null;
  crps_before: number | null;
  crps_after: number | null;
  cov90_before: number | null;
  cov90_after: number | null;
  pit_ks_before: number | null;
  pit_ks_after: number | null;
}

export interface CalibrationResponse {
  stat_types: CalibrationEntry[];
}

export interface EnsembleContext {
  context_key: string;
  stat_type: string;
  regime: string;
  neff_bucket: string;
  weights: Record<string, number>;
}

export interface EnsembleWeightsResponse {
  experts: string[];
  contexts: EnsembleContext[];
}

export interface ConfidenceBin {
  range_start: number;
  range_end: number;
  count: number;
  hits: number | null;
  misses: number | null;
  hit_rate: number | null;
}

export interface ConfidenceDistResponse {
  bins: ConfidenceBin[];
}

export async function fetchTrainingHistory(): Promise<TrainingHistoryResponse> {
  const res = await fetch(`${API_URL}/api/stats/training-history`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchExpertComparison(): Promise<ExpertComparisonResponse> {
  const res = await fetch(`${API_URL}/api/stats/expert-comparison`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHitRate(window: number = 50): Promise<HitRateResponse> {
  const res = await fetch(`${API_URL}/api/stats/hit-rate?window=${window}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchCalibration(): Promise<CalibrationResponse> {
  const res = await fetch(`${API_URL}/api/stats/calibration`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchEnsembleWeights(): Promise<EnsembleWeightsResponse> {
  const res = await fetch(`${API_URL}/api/stats/ensemble-weights`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchConfidenceDist(bins: number = 20): Promise<ConfidenceDistResponse> {
  const res = await fetch(`${API_URL}/api/stats/confidence-dist?bins=${bins}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// --- Weight History & Drift ---

export interface WeightHistoryEntry {
  timestamp: string;
  n_updates: number;
  hedge_avg?: Record<string, number>;
  thompson_avg?: Record<string, number>;
  mixing?: Record<string, number>;
}

export interface WeightHistoryResponse {
  entries: WeightHistoryEntry[];
}

export interface DriftCheck {
  check_type: string;
  is_drifted: boolean;
  metric_value: number;
  threshold: number;
  details: Record<string, unknown>;
}

export interface DriftReportResponse {
  recent_rows: number;
  baseline_rows: number;
  checks: DriftCheck[];
  any_drift: boolean;
}

export interface MixingWeightsResponse {
  mixing: Record<string, number>;
  hedge_avg?: Record<string, number>;
  thompson_avg?: Record<string, number>;
  timestamp?: string;
  n_updates: number;
}

export async function fetchWeightHistory(limit: number = 100): Promise<WeightHistoryResponse> {
  const res = await fetch(`${API_URL}/api/stats/weight-history?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchDriftReport(): Promise<DriftReportResponse> {
  const res = await fetch(`${API_URL}/api/stats/drift-report`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchMixingWeights(): Promise<MixingWeightsResponse> {
  const res = await fetch(`${API_URL}/api/stats/mixing-weights`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// --- Model Health ---

export interface InversionTest {
  accuracy: number;
  accuracy_inverted: number;
  logloss: number;
  logloss_inverted: number;
  inversion_improves_accuracy: boolean;
  inversion_improves_logloss: boolean;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ExpertMetric {
  rolling_accuracy: number;
  rolling_logloss: number;
  n: number;
  ensemble_weight: number;
  alert_eligible: boolean;
  base_rate: number;
  inversion_test?: InversionTest;
  confusion_matrix?: ConfusionMatrix;
}

export interface TierMetric {
  n: number;
  accuracy: number | null;
  threshold?: number;
}

export interface TierMetrics {
  scored: TierMetric;
  actionable: TierMetric;
  placed: TierMetric;
  coverage: number;
  placed_coverage: number;
}

export interface ModelHealthResponse {
  metrics_version: string;
  generated_at: string;
  days_back: number;
  base_rate: number | null;
  tier_metrics: TierMetrics;
  expert_metrics: Record<string, ExpertMetric>;
  alert_count: number;
  suppressed_alert_count: number;
}

export async function fetchModelHealth(
  daysBack: number = 90,
  minAlertWeight: number = 0.03
): Promise<ModelHealthResponse> {
  const res = await fetch(
    `${API_URL}/api/stats/model-health?days_back=${daysBack}&min_alert_weight=${minAlertWeight}`
  );
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
