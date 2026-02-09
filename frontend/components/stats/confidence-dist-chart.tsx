"use client";

import { usePolling } from "@/lib/use-polling";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, LineChart, Line } from "recharts";
import { DataCard } from "@/components/ui/data-card";
import { CHART_MARGINS, CHART_GRID } from "@/lib/constants";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { fetchConfidenceDist } from "@/lib/api";

const histConfig = {
  hits: { label: "Hits", color: "oklch(0.75 0.12 75)" },
  misses: { label: "Misses", color: "oklch(0.4 0 0)" },
  count: { label: "Predictions", color: "oklch(0.75 0.12 75)" },
} satisfies ChartConfig;

const curveConfig = {
  hit_rate: { label: "Hit Rate", color: "oklch(0.75 0.12 75)" },
} satisfies ChartConfig;

export function ConfidenceDistChart() {
  const { data, loading } = usePolling(() => fetchConfidenceDist(20));
  const noData = !data || data.bins.length === 0;
  const hasOutcomes = data?.bins.some((b) => b.hits != null) ?? false;

  const chartData = data?.bins.map((b) => ({
    label: `${(b.range_start * 100).toFixed(0)}–${(b.range_end * 100).toFixed(0)}%`,
    hits: b.hits ?? 0,
    misses: b.misses ?? 0,
    count: b.count,
    hit_rate: b.hit_rate != null ? +(b.hit_rate * 100).toFixed(1) : null,
    midpoint: +((b.range_start + b.range_end) / 2 * 100).toFixed(1),
  })) ?? [];

  return (
    <DataCard
      title="Confidence Distribution"
      description={hasOutcomes
        ? "Stacked by hits/misses with per-bin calibration curve"
        : "Distribution of prediction confidence levels"}
      loading={loading}
      noData={noData}
      noDataDescription="No prediction data available yet."
    >
        <div className="space-y-8">
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Count by confidence bin
            </p>
            <ChartContainer config={histConfig} className="min-h-[240px] w-full">
              <BarChart data={chartData} accessibilityLayer margin={CHART_MARGINS}>
              <CartesianGrid {...CHART_GRID} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} tickMargin={8} angle={-30} textAnchor="end" height={50} interval={0} />
                <YAxis tickLine={false} axisLine={false} tickMargin={4} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {hasOutcomes ? (
                  <>
                    <Bar dataKey="hits" stackId="stack" fill="var(--color-hits)" />
                    <Bar dataKey="misses" stackId="stack" fill="var(--color-misses)" radius={[4, 4, 0, 0]} />
                  </>
                ) : (
                  <Bar dataKey="count" fill="var(--color-count)" radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            </ChartContainer>
          </div>

          {hasOutcomes && (
            <div>
              <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Calibration curve
              </p>
              <ChartContainer config={curveConfig} className="min-h-[220px] w-full">
                <LineChart data={chartData} accessibilityLayer margin={CHART_MARGINS}>
                <CartesianGrid {...CHART_GRID} />
                  <XAxis dataKey="midpoint" tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v: number) => `${v}%`} />
                  <YAxis domain={[30, 100]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v: number) => `${v}%`} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line
                    type="natural"
                    dataKey="hit_rate"
                    stroke="var(--color-hit_rate)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                    connectNulls
                  />
                </LineChart>
              </ChartContainer>
            </div>
          )}
        </div>
    </DataCard>
  );
}
