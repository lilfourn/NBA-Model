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
import { fetchModelHealth } from "@/lib/api";
import type { ExpertMetric } from "@/lib/api";
import { AlertTriangle, Activity, Shield, Target } from "lucide-react";

function TierPill({
  label,
  n,
  accuracy,
  icon: Icon,
}: {
  label: string;
  n: number;
  accuracy: number | null;
  icon: React.ElementType;
}) {
  const accPct = accuracy !== null ? (accuracy * 100).toFixed(1) : "--";
  const color =
    accuracy !== null && accuracy >= 0.55
      ? "text-green-500"
      : accuracy !== null && accuracy >= 0.50
        ? "text-yellow-500"
        : "text-red-400";

  return (
    <div className="flex items-center gap-3 rounded-lg border border-border/50 bg-card px-4 py-3">
      <Icon className="h-4 w-4 text-muted-foreground" />
      <div className="flex-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-lg font-semibold tabular-nums ${color}`}>
          {accPct}%
        </p>
      </div>
      <span className="rounded-full bg-muted px-2 py-0.5 text-xs tabular-nums text-muted-foreground">
        n={n.toLocaleString()}
      </span>
    </div>
  );
}

function InversionRow({
  name,
  metric,
}: {
  name: string;
  metric: ExpertMetric;
}) {
  const inv = metric.inversion_test;
  if (!inv) return null;
  const both = inv.inversion_improves_accuracy && inv.inversion_improves_logloss;
  if (!both) return null;

  const accNormal = (inv.accuracy * 100).toFixed(1);
  const accInv = (inv.accuracy_inverted * 100).toFixed(1);
  const label = name.replace("p_", "").toUpperCase();
  const weight = (metric.ensemble_weight * 100).toFixed(1);

  return (
    <div className="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-2.5">
      <div className="flex items-center gap-3">
        <AlertTriangle className="h-4 w-4 text-amber-500" />
        <div>
          <p className="text-sm font-medium">{label}</p>
          <p className="text-xs text-muted-foreground">
            {accNormal}% acc → {accInv}% if flipped &middot; {weight}% weight
          </p>
        </div>
      </div>
      <span className="rounded-full bg-amber-500/10 px-2.5 py-0.5 text-xs font-medium text-amber-500">
        Auto-corrected
      </span>
    </div>
  );
}

export function ModelHealthCard() {
  const { data, loading } = usePolling(() => fetchModelHealth(), 120_000);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-48" />
          <Skeleton className="mt-1 h-4 w-72" />
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-3">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Health</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No model health data available.
          </p>
        </CardContent>
      </Card>
    );
  }

  const tiers = data.tier_metrics;
  const baseRate = data.base_rate !== null ? (data.base_rate * 100).toFixed(1) : "--";

  // Collect inverted experts
  const invertedExperts = Object.entries(data.expert_metrics || {}).filter(
    ([, m]) => {
      const inv = m.inversion_test;
      return inv?.inversion_improves_accuracy && inv?.inversion_improves_logloss;
    }
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Model Health</CardTitle>
            <CardDescription>
              Tier metrics, inversion status, and alerts &middot; base rate {baseRate}%
            </CardDescription>
          </div>
          {data.alert_count > 0 && (
            <span className="flex items-center gap-1.5 rounded-full bg-red-500/10 px-3 py-1 text-xs font-medium text-red-500">
              <AlertTriangle className="h-3.5 w-3.5" />
              {data.alert_count} alert{data.alert_count !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Tier metrics */}
        <div className="grid gap-3 sm:grid-cols-3">
          <TierPill
            label="Scored"
            n={tiers?.scored?.n ?? 0}
            accuracy={tiers?.scored?.accuracy ?? null}
            icon={Activity}
          />
          <TierPill
            label="Actionable"
            n={tiers?.actionable?.n ?? 0}
            accuracy={tiers?.actionable?.accuracy ?? null}
            icon={Target}
          />
          <TierPill
            label="Placed"
            n={tiers?.placed?.n ?? 0}
            accuracy={tiers?.placed?.accuracy ?? null}
            icon={Shield}
          />
        </div>

        {/* Coverage */}
        {tiers && (
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span>
              Actionable coverage: {((tiers.coverage ?? 0) * 100).toFixed(1)}%
            </span>
            <span>
              Placed coverage: {((tiers.placed_coverage ?? 0) * 100).toFixed(1)}%
            </span>
          </div>
        )}

        {/* Inversion warnings */}
        {invertedExperts.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Inversion Corrections ({invertedExperts.length})
            </p>
            {invertedExperts.map(([name, metric]) => (
              <InversionRow key={name} name={name} metric={metric} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
