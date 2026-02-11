"use client";

import { useCallback } from "react";
import { Line, LineChart, XAxis, YAxis, CartesianGrid } from "recharts";
import { usePolling } from "@/lib/use-polling";
import { DataCard } from "@/components/ui/data-card";
import { fetchCoverageFrontier } from "@/lib/api";
import { CHART_GRID, CHART_MARGINS } from "@/lib/constants";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";

const config = {
  accuracy_actionable: { label: "Actionable Acc", color: "oklch(0.72 0.16 145)" },
  accuracy_placed: { label: "Placed Acc", color: "oklch(0.67 0.13 235)" },
  coverage: { label: "Coverage", color: "oklch(0.66 0.07 80)" },
} satisfies ChartConfig;

export function CoverageFrontierChart() {
  const fetcher = useCallback(() => fetchCoverageFrontier(120), []);
  const { data, loading, error } = usePolling(fetcher, 120_000);

  const chartData = (data?.points ?? []).map((p) => ({
    threshold: p.threshold,
    accuracy_actionable:
      p.accuracy_actionable != null ? +(p.accuracy_actionable * 100).toFixed(2) : null,
    accuracy_placed:
      p.accuracy_placed != null ? +(p.accuracy_placed * 100).toFixed(2) : null,
    coverage: p.coverage != null ? +(p.coverage * 100).toFixed(2) : null,
  }));

  const noData = chartData.length === 0;

  return (
    <DataCard
      title="Coverage Frontier"
      description="Quality vs quantity across confidence thresholds"
      loading={loading}
      noData={noData}
      noDataDescription="No coverage frontier data available yet."
      error={error ? "Failed to refresh coverage frontier." : undefined}
      errorDescription={error?.message}
      skeletonHeight="h-80"
    >
      <ChartContainer config={config} className="min-h-[320px] w-full">
        <LineChart data={chartData} margin={CHART_MARGINS} accessibilityLayer>
          <CartesianGrid {...CHART_GRID} />
          <XAxis
            dataKey="threshold"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
          />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend content={<ChartLegendContent />} />
          <Line
            type="monotone"
            dataKey="accuracy_actionable"
            stroke="var(--color-accuracy_actionable)"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="accuracy_placed"
            stroke="var(--color-accuracy_placed)"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="coverage"
            stroke="var(--color-coverage)"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ChartContainer>
    </DataCard>
  );
}
