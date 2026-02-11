"use client";

import { usePolling } from "@/lib/use-polling";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { fetchExpertComparison } from "@/lib/api";
import { MODEL_META } from "@/lib/constants";

const chartConfig: ChartConfig = {
  value: { label: "Value" },
};

export function ExpertComparisonChart() {
  const { data: resp, loading } = usePolling(fetchExpertComparison);
  const experts = resp?.experts ?? [];

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Expert Comparison</CardTitle></CardHeader>
        <CardContent><Skeleton className="h-72 w-full" /></CardContent>
      </Card>
    );
  }

  if (experts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Expert Comparison</CardTitle>
          <CardDescription>No model runs available yet.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const accData = experts.map((e) => ({
    name: MODEL_META[e.model_name]?.label ?? e.model_name,
    value: e.accuracy ? +(e.accuracy * 100).toFixed(1) : 0,
    fill: MODEL_META[e.model_name]?.color ?? "hsl(var(--chart-5))",
  }));

  const aucData = experts.map((e) => ({
    name: MODEL_META[e.model_name]?.label ?? e.model_name,
    value: e.roc_auc ? +e.roc_auc.toFixed(4) : 0,
    fill: MODEL_META[e.model_name]?.color ?? "hsl(var(--chart-5))",
  }));

  const conformalData = experts.map((e) => ({
    name: MODEL_META[e.model_name]?.label ?? e.model_name,
    value: e.conformal_q_hat ? +e.conformal_q_hat.toFixed(4) : 0,
    fill: MODEL_META[e.model_name]?.color ?? "hsl(var(--chart-5))",
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Expert Comparison</CardTitle>
        <CardDescription>Latest metrics for each model expert</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Summary stat cards */}
        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {experts.map((e) => {
            const meta = MODEL_META[e.model_name];
            return (
              <div
                key={e.model_name}
                className="rounded-lg border border-border bg-white/[0.02] p-3 text-center"
              >
                <p className="text-[11px] font-medium text-muted-foreground">{meta?.label ?? e.model_name}</p>
                <p className="text-2xl font-bold tabular-nums tracking-tight text-primary">
                  {e.accuracy ? `${(e.accuracy * 100).toFixed(1)}%` : "—"}
                </p>
                <p className="text-[11px] text-muted-foreground tabular-nums">
                  AUC {e.roc_auc?.toFixed(3) ?? "—"} · {e.train_rows.toLocaleString()} rows
                </p>
              </div>
            );
          })}
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Accuracy %</p>
            <ChartContainer config={chartConfig} className="min-h-[180px] w-full">
              <BarChart data={accData} accessibilityLayer margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} />
                <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tickMargin={4} tickFormatter={(v: number) => `${v}%`} />
                <ChartTooltip content={<ChartTooltipContent hideIndicator />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </div>

          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">ROC AUC</p>
            <ChartContainer config={chartConfig} className="min-h-[180px] w-full">
              <BarChart data={aucData} accessibilityLayer margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} />
                <YAxis domain={[0.5, 1]} tickLine={false} axisLine={false} tickMargin={4} />
                <ChartTooltip content={<ChartTooltipContent hideIndicator />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </div>

          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Conformal q̂</p>
            <ChartContainer config={chartConfig} className="min-h-[180px] w-full">
              <BarChart data={conformalData} accessibilityLayer margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} />
                <YAxis domain={[0, 1]} tickLine={false} axisLine={false} tickMargin={4} />
                <ChartTooltip content={<ChartTooltipContent hideIndicator />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
