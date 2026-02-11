"use client";

import { usePolling } from "@/lib/use-polling";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { fetchWeightHistory } from "@/lib/api";
import type { WeightHistoryEntry } from "@/lib/api";
import { EXPERT_COLORS, EXPERT_LABELS } from "@/lib/constants";

function formatDate(ts: string): string {
  try {
    const d = new Date(ts);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  } catch {
    return ts.slice(0, 10);
  }
}

export function WeightHistoryChart() {
  const { data, loading } = usePolling(fetchWeightHistory);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Weight Evolution</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-72 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!data || data.entries.length < 2) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Weight Evolution</CardTitle>
          <CardDescription>
            Not enough weight history data yet. Weights are logged after each
            ensemble training run.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const entries = data.entries;

  // Build line data for hedge weights
  const experts = new Set<string>();
  for (const e of entries) {
    if (e.hedge_avg) Object.keys(e.hedge_avg).forEach((k) => experts.add(k));
  }

  const chartData = entries.map((e: WeightHistoryEntry) => {
    const row: Record<string, unknown> = {
      date: formatDate(e.timestamp),
      n_updates: e.n_updates,
    };
    for (const expert of experts) {
      row[expert] = e.hedge_avg?.[expert] ?? null;
    }
    return row;
  });

  const chartConfig: ChartConfig = {};
  for (const expert of experts) {
    chartConfig[expert] = {
      label: EXPERT_LABELS[expert] ?? expert,
      color: EXPERT_COLORS[expert] ?? "hsl(var(--chart-5))",
    };
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Weight Evolution</CardTitle>
        <CardDescription>
          Hedge ensemble weights over {entries.length} training runs
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="min-h-[320px] w-full">
          <LineChart
            data={chartData}
            accessibilityLayer
            margin={{ top: 5, right: 12, bottom: 10, left: 0 }}
          >
            <CartesianGrid vertical={false} strokeOpacity={0.06} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={6}
              fontSize={10}
            />
            <YAxis
              domain={[0, "auto"]}
              tickLine={false}
              axisLine={false}
              tickMargin={4}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {[...experts].map((expert) => (
              <Line
                key={expert}
                type="monotone"
                dataKey={expert}
                stroke={`var(--color-${expert})`}
                strokeWidth={2}
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
