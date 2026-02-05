"use client";

import { usePolling } from "@/lib/use-polling";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchDriftReport } from "@/lib/api";
import type { DriftCheck } from "@/lib/api";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

function DriftCheckRow({ check }: { check: DriftCheck }) {
  const pct = (check.metric_value * 100).toFixed(1);
  const threshPct = (check.threshold * 100).toFixed(1);

  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-4 py-3 ${
        check.is_drifted
          ? "border-red-500/30 bg-red-500/5"
          : "border-green-500/20 bg-green-500/5"
      }`}
    >
      <div className="flex items-center gap-3">
        {check.is_drifted ? (
          <AlertTriangle className="h-4 w-4 text-red-500" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        )}
        <div>
          <p className="text-sm font-medium capitalize">
            {check.check_type.replace("_", " ")}
          </p>
          <p className="text-xs text-muted-foreground">
            {pct}% (threshold: {threshPct}%)
          </p>
        </div>
      </div>
      <span
        className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
          check.is_drifted
            ? "bg-red-500/10 text-red-500"
            : "bg-green-500/10 text-green-500"
        }`}
      >
        {check.is_drifted ? "Drift" : "OK"}
      </span>
    </div>
  );
}

export function DriftStatusCard() {
  const { data, loading } = usePolling(fetchDriftReport);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Drift Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!data || data.checks.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Drift Detection</CardTitle>
          <CardDescription>
            No drift checks available yet. Run the drift detection job.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Drift Detection
          {data.any_drift && (
            <span className="rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-500">
              Drift Detected
            </span>
          )}
        </CardTitle>
        <CardDescription>
          {data.recent_rows} recent / {data.baseline_rows} baseline predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {data.checks.map((check) => (
            <DriftCheckRow key={check.check_type} check={check} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
