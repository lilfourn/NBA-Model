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
  p_lr: number | null;
  p_xgb: number | null;
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

export async function fetchPicks(params?: {
  snapshot_id?: string;
  game_date?: string;
  top?: number;
  rank?: string;
  include_non_today?: boolean;
}): Promise<ScoringResult> {
  const searchParams = new URLSearchParams();
  if (params?.snapshot_id) searchParams.set("snapshot_id", params.snapshot_id);
  if (params?.game_date) searchParams.set("game_date", params.game_date);
  if (params?.top) searchParams.set("top", params.top.toString());
  if (params?.rank) searchParams.set("rank", params.rank);
  if (params?.include_non_today)
    searchParams.set("include_non_today", "true");

  const url = `${API_URL}/api/picks?${searchParams.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
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
