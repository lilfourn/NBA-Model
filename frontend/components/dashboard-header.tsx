"use client";

import { Input } from "@/components/ui/input";
import { SnapshotSelector } from "@/components/snapshot-selector";
import { StatTypeFilter } from "@/components/stat-type-filter";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Activity } from "lucide-react";
import type { Snapshot } from "@/lib/api";

interface DashboardHeaderProps {
  snapshots: Snapshot[];
  selectedSnapshot: string;
  onSnapshotChange: (id: string) => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  statTypes: string[];
  statTypeFilter: string;
  onStatTypeChange: (v: string) => void;
  topN: number;
  onTopNChange: (n: number) => void;
  rankStrategy: string;
  onRankStrategyChange: (v: string) => void;
  totalScored: number;
  isLoading: boolean;
}

export function DashboardHeader({
  snapshots,
  selectedSnapshot,
  onSnapshotChange,
  searchQuery,
  onSearchChange,
  statTypes,
  statTypeFilter,
  onStatTypeChange,
  topN,
  onTopNChange,
  rankStrategy,
  onRankStrategyChange,
  totalScored,
  isLoading,
}: DashboardHeaderProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-7 w-7 text-primary" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              NBA Picks Dashboard
            </h1>
            <p className="text-sm text-muted-foreground">
              Ensemble model predictions
              {totalScored > 0 && !isLoading && (
                <span> · {totalScored} projections scored</span>
              )}
            </p>
          </div>
        </div>
        <SnapshotSelector
          snapshots={snapshots}
          value={selectedSnapshot}
          onChange={onSnapshotChange}
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search player..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9 w-[220px]"
          />
        </div>

        <StatTypeFilter
          statTypes={statTypes}
          value={statTypeFilter}
          onChange={onStatTypeChange}
        />

        <Select
          value={topN.toString()}
          onValueChange={(v) => onTopNChange(Number(v))}
        >
          <SelectTrigger className="w-[100px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[10, 25, 50, 100].map((n) => (
              <SelectItem key={n} value={n.toString()}>
                Top {n}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={rankStrategy} onValueChange={onRankStrategyChange}>
          <SelectTrigger className="w-[150px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="risk_adj">Risk Adjusted</SelectItem>
            <SelectItem value="confidence">Confidence</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
