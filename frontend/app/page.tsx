"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { PicksTable } from "@/components/picks-table";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchPicks, fetchSnapshots, getCachedPicks } from "@/lib/api";
import type { ScoredPick, Snapshot } from "@/lib/api";
import { AlertCircle, Loader2 } from "lucide-react";

export default function Home() {
  const [picks, setPicks] = useState<ScoredPick[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState("latest");
  const [searchQuery, setSearchQuery] = useState("");
  const [statTypeFilter, setStatTypeFilter] = useState("all");
  const [topN, setTopN] = useState(50);
  const [rankStrategy, setRankStrategy] = useState("risk_adj");
  const [totalScored, setTotalScored] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isScoring, setIsScoring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialLoadDone = useRef(false);

  const statTypes = useMemo(() => {
    const set = new Set(picks.map((p) => p.stat_type));
    return Array.from(set).sort();
  }, [picks]);

  const loadPicks = useCallback(
    async (force = false) => {
      if (force) setIsScoring(true);
      if (picks.length === 0) setIsLoading(true);
      setError(null);
      try {
        const result = await fetchPicks({
          snapshot_id:
            selectedSnapshot === "latest" ? undefined : selectedSnapshot,
          top: topN,
          rank: rankStrategy,
          force,
        });
        setPicks(result.picks);
        setTotalScored(result.total_scored);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load picks");
        if (picks.length === 0) setPicks([]);
      } finally {
        setIsLoading(false);
        setIsScoring(false);
      }
    },
    [selectedSnapshot, topN, rankStrategy, picks.length],
  );

  const handleRerun = useCallback(() => {
    loadPicks(true);
  }, [loadPicks]);

  const loadSnapshots = useCallback(async () => {
    try {
      const result = await fetchSnapshots();
      setSnapshots(result.snapshots);
    } catch {
      // Non-critical: snapshot list just won't populate
    }
  }, []);

  // On mount: hydrate from localStorage cache instantly
  useEffect(() => {
    if (initialLoadDone.current) return;
    initialLoadDone.current = true;
    const cached = getCachedPicks();
    if (cached && cached.picks.length > 0) {
      setPicks(cached.picks);
      setTotalScored(cached.total_scored);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSnapshots();
  }, [loadSnapshots]);

  useEffect(() => {
    loadPicks();
  }, [loadPicks]);

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <DashboardHeader
          snapshots={snapshots}
          selectedSnapshot={selectedSnapshot}
          onSnapshotChange={setSelectedSnapshot}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          statTypes={statTypes}
          statTypeFilter={statTypeFilter}
          onStatTypeChange={setStatTypeFilter}
          topN={topN}
          onTopNChange={setTopN}
          rankStrategy={rankStrategy}
          onRankStrategyChange={setRankStrategy}
          totalScored={totalScored}
          isLoading={isLoading}
          onRerun={handleRerun}
          isScoring={isScoring}
        />

        <div className="mt-6">
          {isLoading ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Running ensemble scoring...</span>
              </div>
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : error ? (
            <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <div>
                <p className="font-medium text-destructive">
                  Failed to load picks
                </p>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            </div>
          ) : (
            <PicksTable
              picks={picks}
              searchQuery={searchQuery}
              statTypeFilter={statTypeFilter}
            />
          )}
        </div>
      </div>
    </div>
  );
}
