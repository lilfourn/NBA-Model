"use client";

import { useCallback } from "react";
import { usePolling } from "@/lib/use-polling";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine } from "recharts";
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
import { fetchHitRate } from "@/lib/api";

const chartConfig = {
  ensemble_hit_rate: { label: "Ensemble", color: "oklch(0.75 0.12 75)" },
  p_forecast_cal_hit_rate: { label: "Forecast", color: "oklch(0.55 0 0)" },
  p_nn_hit_rate: { label: "NN", color: "oklch(0.45 0 0)" },
  p_lr_hit_rate: { label: "LR", color: "oklch(0.5 0.03 180)" },
  p_xgb_hit_rate: { label: "XGB", color: "oklch(0.5 0.04 250)" },
} satisfies ChartConfig;

const EXPERT_KEYS = ["p_forecast_cal_hit_rate", "p_nn_hit_rate", "p_lr_hit_rate", "p_xgb_hit_rate"] as const;

export function HitRateChart() {
  const fetcher = useCallback(() => fetchHitRate(50), []);
  const { data, loading } = usePolling(fetcher);

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Prediction Hit Rate</CardTitle></CardHeader>
        <CardContent><Skeleton className="h-80 w-full" /></CardContent>
      </Card>
    );
  }

  if (!data || data.rolling.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction Hit Rate</CardTitle>
          <CardDescription>No resolved predictions yet. Outcomes will appear after games complete.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const chartData = data.rolling.map((p) => ({
    ...p,
    ensemble_hit_rate: p.ensemble_hit_rate != null ? +(p.ensemble_hit_rate * 100).toFixed(1) : null,
    p_forecast_cal_hit_rate: p.p_forecast_cal_hit_rate != null ? +(p.p_forecast_cal_hit_rate * 100).toFixed(1) : null,
    p_nn_hit_rate: p.p_nn_hit_rate != null ? +(p.p_nn_hit_rate * 100).toFixed(1) : null,
    p_lr_hit_rate: p.p_lr_hit_rate != null ? +(p.p_lr_hit_rate * 100).toFixed(1) : null,
    p_xgb_hit_rate: p.p_xgb_hit_rate != null ? +(p.p_xgb_hit_rate * 100).toFixed(1) : null,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction Hit Rate</CardTitle>
        <CardDescription>Rolling 50-prediction window accuracy</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Summary cards */}
        <div className="mb-6 grid grid-cols-3 gap-3">
          <div className="rounded-lg border border-border bg-white/[0.02] p-3 text-center">
            <p className="text-[11px] font-medium text-muted-foreground">Overall Hit Rate</p>
            <p className="text-2xl font-bold tabular-nums tracking-tight text-primary">
              {data.overall_hit_rate != null ? `${(data.overall_hit_rate * 100).toFixed(1)}%` : "—"}
            </p>
          </div>
          <div className="rounded-lg border border-border bg-white/[0.02] p-3 text-center">
            <p className="text-[11px] font-medium text-muted-foreground">Total Predictions</p>
            <p className="text-2xl font-bold tabular-nums tracking-tight">{data.total_predictions.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-border bg-white/[0.02] p-3 text-center">
            <p className="text-[11px] font-medium text-muted-foreground">Resolved</p>
            <p className="text-2xl font-bold tabular-nums tracking-tight">{data.total_resolved.toLocaleString()}</p>
          </div>
        </div>

        <ChartContainer config={chartConfig} className="min-h-[300px] w-full">
          <LineChart data={chartData} accessibilityLayer margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.06} />
            <XAxis dataKey="index" tickLine={false} axisLine={false} tickMargin={8} />
            <YAxis domain={[30, 80]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v: number) => `${v}%`} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <ReferenceLine y={50} strokeDasharray="6 3" strokeOpacity={0.15} />
            <Line
              type="natural"
              dataKey="ensemble_hit_rate"
              stroke="var(--color-ensemble_hit_rate)"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            {EXPERT_KEYS.map((key) => (
              <Line
                key={key}
                type="natural"
                dataKey={key}
                stroke={`var(--color-${key})`}
                strokeWidth={1.5}
                strokeOpacity={0.5}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
