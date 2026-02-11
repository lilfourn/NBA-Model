"use client";

import { useCallback } from "react";
import { usePolling } from "@/lib/use-polling";
import { DataCard } from "@/components/ui/data-card";
import { fetchPerStatPerformance } from "@/lib/api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function pct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function PerStatPerformanceCard() {
  const fetcher = useCallback(() => fetchPerStatPerformance(120), []);
  const { data, loading, error } = usePolling(fetcher, 120_000);

  const rows = data?.stats ?? [];
  const noData = rows.length === 0;

  return (
    <DataCard
      title="Per-Stat Performance"
      description="Scored vs actionable vs placed quality by stat type"
      loading={loading}
      noData={noData}
      noDataDescription="No per-stat performance rows available yet."
      error={error ? "Failed to refresh per-stat performance." : undefined}
      errorDescription={error?.message}
      skeletonHeight="h-80"
    >
      <div className="max-h-[420px] overflow-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Stat</TableHead>
              <TableHead className="text-right">Scored</TableHead>
              <TableHead className="text-right">Scored Acc</TableHead>
              <TableHead className="text-right">Actionable</TableHead>
              <TableHead className="text-right">Actionable Acc</TableHead>
              <TableHead className="text-right">Placed Acc</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.stat_type}>
                <TableCell className="font-medium">{row.stat_type}</TableCell>
                <TableCell className="text-right tabular-nums">{row.n_scored}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {pct(row.accuracy_scored)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{row.n_actionable}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {pct(row.accuracy_actionable)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {pct(row.accuracy_placed)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </DataCard>
  );
}
