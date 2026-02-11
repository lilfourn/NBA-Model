"use client";

import { useState } from "react";
import { usePolling } from "@/lib/use-polling";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell } from "recharts";
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
import { fetchEnsembleWeights } from "@/lib/api";
import type { EnsembleContext } from "@/lib/api";
import { abbrevStat, EXPERT_META } from "@/lib/constants";

export function EnsembleWeightsChart() {
  const { data, loading } = usePolling(fetchEnsembleWeights);
  const [selectedContextKey, setSelectedContextKey] = useState<string>("");

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Ensemble Weights</CardTitle></CardHeader>
        <CardContent><Skeleton className="h-72 w-full" /></CardContent>
      </Card>
    );
  }

  if (!data || data.contexts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Ensemble Weights</CardTitle>
          <CardDescription>No ensemble weights trained yet. Run the ensemble training job.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const defaultContext = data.contexts[0];
  const selected: EnsembleContext =
    data.contexts.find((ctx) => ctx.context_key === selectedContextKey) ?? defaultContext;

  // Build chart config dynamically from the experts in the data
  const barConfig: ChartConfig = {};
  for (const expert of data.experts) {
    const meta = EXPERT_META[expert] ?? { label: expert, color: "hsl(var(--chart-5))" };
    barConfig[expert] = { label: meta.label, color: meta.color };
  }

  const statTypes = [...new Set(data.contexts.map((c) => c.stat_type))];

  const barData = statTypes.map((st) => {
    const ctxs = data.contexts.filter((c) => c.stat_type === st);
    const rawAvg: Record<string, number> = {};
    for (const expert of data.experts) {
      const vals = ctxs.map((c) => c.weights[expert] ?? 0);
      rawAvg[expert] = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
    const total = Object.values(rawAvg).reduce((a, b) => a + b, 0) || 1;
    const entry: Record<string, unknown> = { stat_type: abbrevStat(st) };
    const pcts = data.experts.map((e) => Math.round((rawAvg[e] / total) * 1000) / 10);
    const pctTotal = pcts.reduce((a, b) => a + b, 0);
    const diff = +(100 - pctTotal).toFixed(1);
    if (diff !== 0 && pcts.length > 0) pcts[0] = +(pcts[0] + diff).toFixed(1);
    data.experts.forEach((e, i) => { entry[e] = pcts[i]; });
    return entry;
  });

  const pieConfig: ChartConfig = {};
  const pieData = data.experts.map((e) => {
    const meta = EXPERT_META[e] ?? { label: e, color: "hsl(var(--chart-5))" };
    pieConfig[meta.label] = { label: meta.label, color: meta.color };
    return {
      name: meta.label,
      value: +((selected.weights[e] ?? 0) * 100).toFixed(1),
      fill: meta.color,
    };
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ensemble Weights</CardTitle>
        <CardDescription>
          Online Hedge ensemble allocation — {data.contexts.length} context buckets
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Stacked bar chart */}
          <div className="lg:col-span-2">
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Avg weight % by stat type
            </p>
            <ChartContainer config={barConfig} className="min-h-[320px] w-full">
              <BarChart data={barData} accessibilityLayer margin={{ top: 5, right: 12, bottom: 10, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="stat_type" tickLine={false} axisLine={false} tickMargin={6} angle={-35} textAnchor="end" height={60} interval={0} fontSize={10} />
                <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tickMargin={4} tickFormatter={(v: number) => `${v}%`} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {data.experts.map((expert) => (
                  <Bar
                    key={expert}
                    dataKey={expert}
                    stackId="weights"
                    fill={`var(--color-${expert})`}
                    radius={expert === data.experts[data.experts.length - 1] ? [4, 4, 0, 0] : undefined}
                  />
                ))}
              </BarChart>
            </ChartContainer>
          </div>

          {/* Pie chart for selected context */}
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Context detail
            </p>
            <select
              className="mb-4 w-full rounded-md border border-border bg-white/[0.04] px-2.5 py-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              value={selected.context_key}
              onChange={(e) => setSelectedContextKey(e.target.value)}
            >
              {data.contexts.map((ctx) => (
                <option key={ctx.context_key} value={ctx.context_key}>
                  {abbrevStat(ctx.stat_type)} · {ctx.regime} · {ctx.neff_bucket}
                </option>
              ))}
            </select>

            <ChartContainer config={pieConfig} className="mx-auto min-h-[200px] w-full max-w-[260px]">
              <PieChart>
                <ChartTooltip content={<ChartTooltipContent hideLabel />} />
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  nameKey="name"
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  label={(props: any) => `${props.name ?? ""} ${props.value ?? 0}%`}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
              </PieChart>
            </ChartContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
