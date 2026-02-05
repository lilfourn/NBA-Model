"use client";

import { usePolling } from "@/lib/use-polling";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { fetchTrainingHistory } from "@/lib/api";

const MODEL_KEYS: Record<string, { label: string; color: string }> = {
  baseline_logreg: { label: "Logistic Regression", color: "hsl(var(--chart-1))" },
  nn_gru_attention: { label: "Neural Network", color: "hsl(var(--chart-2))" },
  xgboost: { label: "XGBoost", color: "hsl(var(--chart-3))" },
  lightgbm: { label: "LightGBM", color: "hsl(var(--chart-4))" },
  meta_learner: { label: "Meta Learner", color: "hsl(var(--chart-5))" },
};

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function TrainingHistoryChart() {
  const { data: resp, loading } = usePolling(fetchTrainingHistory);
  const runs = resp?.runs ?? [];

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training History</CardTitle>
        </CardHeader>
        <CardContent><Skeleton className="h-72 w-full" /></CardContent>
      </Card>
    );
  }

  if (runs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training History</CardTitle>
          <CardDescription>No training runs recorded yet.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const modelNames = [...new Set(runs.map((r) => r.model_name))];

  // Build chart config dynamically from discovered model names
  const accConfig: ChartConfig = {};
  const aucConfig: ChartConfig = {};
  for (const name of modelNames) {
    const info = MODEL_KEYS[name] ?? { label: name, color: "hsl(var(--chart-5))" };
    accConfig[`${name}_accuracy`] = { label: info.label, color: info.color };
    aucConfig[`${name}_auc`] = { label: info.label, color: info.color };
  }

  // Pivot: one row per date, columns per model
  const dateMap = new Map<string, Record<string, unknown>>();
  for (const run of runs) {
    const key = formatDate(run.created_at);
    if (!dateMap.has(key)) dateMap.set(key, { date: key });
    const entry = dateMap.get(key)!;
    entry[`${run.model_name}_accuracy`] = run.accuracy != null ? +(run.accuracy * 100).toFixed(1) : null;
    entry[`${run.model_name}_auc`] = run.roc_auc;
    entry[`${run.model_name}_rows`] = run.train_rows;
  }
  const data = Array.from(dateMap.values());

  return (
    <Card>
      <CardHeader>
        <CardTitle>Training History</CardTitle>
        <CardDescription>Model accuracy &amp; AUC across training runs</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Accuracy (%)</p>
            <ChartContainer config={accConfig} className="min-h-[240px] w-full">
              <LineChart data={data} accessibilityLayer margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="date" tickLine={false} axisLine={false} tickMargin={8} />
                <YAxis domain={[40, 100]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v: number) => `${v}%`} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {modelNames.map((name) => (
                  <Line
                    key={name}
                    type="natural"
                    dataKey={`${name}_accuracy`}
                    stroke={`var(--color-${name}_accuracy)`}
                    strokeWidth={2}
                    dot={{ fill: `var(--color-${name}_accuracy)`, r: 4 }}
                    activeDot={{ r: 6 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ChartContainer>
          </div>
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">ROC AUC</p>
            <ChartContainer config={aucConfig} className="min-h-[240px] w-full">
              <LineChart data={data} accessibilityLayer margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="date" tickLine={false} axisLine={false} tickMargin={8} />
                <YAxis domain={[0.5, 1]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v: number) => v.toFixed(2)} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {modelNames.map((name) => (
                  <Line
                    key={name}
                    type="natural"
                    dataKey={`${name}_auc`}
                    stroke={`var(--color-${name}_auc)`}
                    strokeWidth={2}
                    dot={{ fill: `var(--color-${name}_auc)`, r: 4 }}
                    activeDot={{ r: 6 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ChartContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
