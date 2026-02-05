"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchJobs, startJob } from "@/lib/api";
import type { Job } from "@/lib/api";

interface JobButton {
  value: string;
  label: string;
  description: string;
}

const JOB_BUTTONS: JobButton[] = [
  {
    value: "train_baseline",
    label: "Train Baseline (LR)",
    description: "Retrain logistic regression model on latest data",
  },
  {
    value: "train_nn",
    label: "Train Neural Network",
    description: "Retrain GRU+Attention model",
  },
  {
    value: "train_ensemble",
    label: "Train Ensemble",
    description: "Update online Hedge ensemble weights from prediction log",
  },
  {
    value: "build_backtest",
    label: "Build Backtest",
    description: "Generate forecast backtest dataset for calibration",
  },
  {
    value: "calibrate",
    label: "Calibrate Forecast",
    description: "Run MLE + PIT calibration on backtest data",
  },
];

function StatusBadge({ status }: { status: Job["status"] }) {
  switch (status) {
    case "running":
      return (
        <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 gap-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          Running
        </Badge>
      );
    case "completed":
      return (
        <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200 gap-1">
          <CheckCircle2 className="h-3 w-3" />
          Completed
        </Badge>
      );
    case "failed":
      return (
        <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 gap-1">
          <XCircle className="h-3 w-3" />
          Failed
        </Badge>
      );
    default:
      return (
        <Badge variant="secondary" className="gap-1">
          Pending
        </Badge>
      );
  }
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function JobHistoryRow({ job }: { job: Job }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b last:border-b-0">
      <div
        className="flex items-center gap-3 py-2 px-1 cursor-pointer hover:bg-muted/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="text-sm font-medium flex-1 truncate">
          {job.label}
        </span>
        <StatusBadge status={job.status} />
        {job.duration_seconds !== null && (
          <span className="text-xs text-muted-foreground font-mono">
            {formatDuration(job.duration_seconds)}
          </span>
        )}
        {job.started_at && (
          <span className="text-xs text-muted-foreground">
            {formatTime(job.started_at)}
          </span>
        )}
      </div>
      {expanded && (
        <div className="px-6 pb-3">
          {job.error && (
            <p className="text-sm text-destructive mb-2">{job.error}</p>
          )}
          {job.output && (
            <pre className="text-xs bg-muted/50 rounded p-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
              {job.output}
            </pre>
          )}
          {!job.output && !job.error && (
            <p className="text-xs text-muted-foreground">No output yet.</p>
          )}
        </div>
      )}
    </div>
  );
}

export function OperationsPanel() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [runningTypes, setRunningTypes] = useState<Set<string>>(new Set());
  const [startError, setStartError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadJobs = useCallback(async () => {
    try {
      const result = await fetchJobs();
      setJobs(result);
      const running = new Set(
        result.filter((j) => j.status === "running").map((j) => j.job_type)
      );
      setRunningTypes(running);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  // Poll while any job is running
  useEffect(() => {
    if (runningTypes.size > 0) {
      pollRef.current = setInterval(loadJobs, 2000);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [runningTypes.size, loadJobs]);

  async function handleStart(jobType: string) {
    setStartError(null);
    try {
      await startJob(jobType);
      setRunningTypes((prev) => new Set([...prev, jobType]));
      await loadJobs();
    } catch (err) {
      setStartError(
        err instanceof Error ? err.message : "Failed to start job"
      );
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Wrench className="h-5 w-5" />
          Operations
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-2">
          {JOB_BUTTONS.map((jb) => {
            const isRunning = runningTypes.has(jb.value);
            return (
              <button
                key={jb.value}
                onClick={() => handleStart(jb.value)}
                disabled={isRunning}
                className={cn(
                  "flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                <div className="flex items-center gap-2 w-full">
                  {isRunning ? (
                    <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                  ) : (
                    <Play className="h-4 w-4 text-emerald-600" />
                  )}
                  <span className="text-sm font-medium">{jb.label}</span>
                </div>
                <span className="text-xs text-muted-foreground leading-tight">
                  {jb.description}
                </span>
              </button>
            );
          })}
        </div>

        {startError && (
          <p className="text-sm text-destructive">{startError}</p>
        )}

        {jobs.length > 0 && (
          <>
            <Separator />
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-2">
                Job History
              </h4>
              <div className="rounded-lg border">
                {jobs.map((job) => (
                  <JobHistoryRow key={job.id} job={job} />
                ))}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
