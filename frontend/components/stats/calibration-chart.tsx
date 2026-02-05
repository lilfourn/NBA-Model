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
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { fetchCalibration } from "@/lib/api";

const nllConfig = {
  before: { label: "Before", color: "oklch(0.45 0 0)" },
  after: { label: "After", color: "oklch(0.75 0.12 75)" },
} satisfies ChartConfig;

const covConfig = {
  before: { label: "Before", color: "oklch(0.45 0 0)" },
  after: { label: "After", color: "oklch(0.75 0.12 75)" },
} satisfies ChartConfig;

export function CalibrationChart() {
  const { data: resp, loading } = usePolling(fetchCalibration);
  const entries = resp?.stat_types ?? [];

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Calibration Quality</CardTitle></CardHeader>
        <CardContent><Skeleton className="h-72 w-full" /></CardContent>
      </Card>
    );
  }

  if (entries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Calibration Quality</CardTitle>
          <CardDescription>No calibration reports found. Run the calibration job first.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const nllData = entries.map((e) => ({
    stat_type: e.stat_type === "__global__" ? "Global" : e.stat_type,
    before: e.nll_before != null ? +e.nll_before.toFixed(4) : 0,
    after: e.nll_after != null ? +e.nll_after.toFixed(4) : 0,
  }));

  const coverageData = entries.map((e) => ({
    stat_type: e.stat_type === "__global__" ? "Global" : e.stat_type,
    before: e.cov90_before != null ? +(e.cov90_before * 100).toFixed(1) : 0,
    after: e.cov90_after != null ? +(e.cov90_after * 100).toFixed(1) : 0,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Calibration Quality</CardTitle>
        <CardDescription>Before vs. after — lower NLL is better, coverage target is 90%</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Negative Log-Likelihood
            </p>
            <ChartContainer config={nllConfig} className="min-h-[220px] w-full">
              <BarChart data={nllData} accessibilityLayer margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="stat_type" tickLine={false} axisLine={false} tickMargin={8} angle={-35} textAnchor="end" height={70} interval={0} />
                <YAxis tickLine={false} axisLine={false} tickMargin={4} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="before" fill="var(--color-before)" radius={[4, 4, 0, 0]} opacity={0.4} />
                <Bar dataKey="after" fill="var(--color-after)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </div>

          <div>
            <p className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              90% Interval Coverage
            </p>
            <ChartContainer config={covConfig} className="min-h-[220px] w-full">
              <BarChart data={coverageData} accessibilityLayer margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
                <CartesianGrid vertical={false} strokeOpacity={0.06} />
                <XAxis dataKey="stat_type" tickLine={false} axisLine={false} tickMargin={8} angle={-35} textAnchor="end" height={70} interval={0} />
                <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tickMargin={4} tickFormatter={(v: number) => `${v}%`} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="before" fill="var(--color-before)" radius={[4, 4, 0, 0]} opacity={0.4} />
                <Bar dataKey="after" fill="var(--color-after)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </div>

          {/* Data table */}
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-white/[0.02] text-left text-muted-foreground">
                  <th className="px-3 py-2">Stat Type</th>
                  <th className="px-3 py-2 text-right">Train</th>
                  <th className="px-3 py-2 text-right">Val</th>
                  <th className="px-3 py-2 text-right">NLL Before</th>
                  <th className="px-3 py-2 text-right">NLL After</th>
                  <th className="px-3 py-2 text-right">CRPS Before</th>
                  <th className="px-3 py-2 text-right">CRPS After</th>
                  <th className="px-3 py-2 text-right">Cov90</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <tr key={e.stat_type} className="border-b border-border hover:bg-white/[0.03] transition-colors">
                    <td className="px-3 py-2 font-medium">{e.stat_type === "__global__" ? "Global" : e.stat_type}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.train_rows?.toLocaleString() ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.val_rows?.toLocaleString() ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.nll_before?.toFixed(4) ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.nll_after?.toFixed(4) ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.crps_before?.toFixed(4) ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.crps_after?.toFixed(4) ?? "—"}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.cov90_after != null ? `${(e.cov90_after * 100).toFixed(1)}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
