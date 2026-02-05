"use client";

import { Fragment, useMemo, useState, type ReactNode } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { EdgeBadge } from "@/components/edge-badge";
import { ExpertBreakdown } from "@/components/expert-breakdown";
import { ChevronDown, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import type { ScoredPick } from "@/lib/api";
import { cn } from "@/lib/utils";

type SortKey =
  | "edge"
  | "prob_over"
  | "line_score"
  | "player_name"
  | "stat_type";

type SortDir = "asc" | "desc";

interface PicksTableProps {
  picks: ScoredPick[];
  searchQuery: string;
  statTypeFilter: string;
}

interface HeaderCellProps {
  column: SortKey;
  children: ReactNode;
  className?: string;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}

function SortIcon({
  column,
  sortKey,
  sortDir,
}: {
  column: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
}) {
  if (sortKey !== column) return <ArrowUpDown className="ml-1 h-3 w-3 opacity-40" />;
  return sortDir === "asc" ? (
    <ArrowUp className="ml-1 h-3 w-3" />
  ) : (
    <ArrowDown className="ml-1 h-3 w-3" />
  );
}

function HeaderCell({
  column,
  children,
  className,
  sortKey,
  sortDir,
  onSort,
}: HeaderCellProps) {
  return (
    <TableHead
      className={cn("cursor-pointer select-none whitespace-nowrap", className)}
      onClick={() => onSort(column)}
    >
      <span className="inline-flex items-center">
        {children}
        <SortIcon column={column} sortKey={sortKey} sortDir={sortDir} />
      </span>
    </TableHead>
  );
}

function playerInitials(name: string): string {
  const initials = name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((segment) => segment[0]?.toUpperCase() || "")
    .join("");
  return initials || "?";
}

export function PicksTable({
  picks,
  searchQuery,
  statTypeFilter,
}: PicksTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("edge");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const filtered = useMemo(() => {
    let result = picks;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((p) => p.player_name.toLowerCase().includes(q));
    }
    if (statTypeFilter && statTypeFilter !== "all") {
      result = result.filter((p) => p.stat_type === statTypeFilter);
    }
    return result;
  }, [picks, searchQuery, statTypeFilter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let av: string | number = a[sortKey] ?? 0;
      let bv: string | number = b[sortKey] ?? 0;
      if (typeof av === "string") av = av.toLowerCase();
      if (typeof bv === "string") bv = bv.toLowerCase();
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return arr;
  }, [filtered, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir(key === "player_name" || key === "stat_type" ? "asc" : "desc");
    }
  }

  if (sorted.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No picks match current filters.
      </div>
    );
  }

  return (
    <div className="rounded-lg border overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="border-b border-border">
            <TableHead className="w-8">#</TableHead>
            <TableHead className="w-8" />
            <HeaderCell
              column="player_name"
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            >
              Player
            </HeaderCell>
            <HeaderCell
              column="stat_type"
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            >
              Stat
            </HeaderCell>
            <HeaderCell
              column="line_score"
              className="text-right"
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            >
              Line
            </HeaderCell>
            <HeaderCell
              column="prob_over"
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            >
              Prediction
            </HeaderCell>
            <HeaderCell
              column="edge"
              className="text-right"
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            >
              Edge
            </HeaderCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((pick, idx) => {
            const isExpanded = expandedId === pick.projection_id;
            return (
              <Fragment key={pick.projection_id}>
                <TableRow
                  className={cn(
                    "cursor-pointer transition-colors hover:bg-white/[0.03]",
                    isExpanded && "bg-white/[0.02]"
                  )}
                  onClick={() =>
                    setExpandedId(isExpanded ? null : pick.projection_id)
                  }
                >
                  <TableCell className="font-mono text-muted-foreground text-sm">
                    {idx + 1}
                  </TableCell>
                  <TableCell className="px-0">
                    {isExpanded ? (
                      <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/50" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
                    )}
                  </TableCell>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-3">
                      {pick.player_image_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={pick.player_image_url}
                          alt={`${pick.player_name} profile`}
                          className="h-8 w-8 rounded-full border object-cover"
                          loading="lazy"
                        />
                      ) : (
                        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-white/[0.04] text-xs font-semibold text-muted-foreground">
                          {playerInitials(pick.player_name)}
                        </div>
                      )}
                      <span>{pick.player_name}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="font-normal">
                      {pick.stat_type}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {pick.line_score.toFixed(1)}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Badge
                        className={cn(
                          "font-semibold text-xs w-14 justify-center border-0",
                          pick.pick === "OVER"
                            ? "bg-emerald-500/15 text-emerald-400"
                            : "bg-red-500/15 text-red-400"
                        )}
                      >
                        {pick.pick}
                      </Badge>
                      <span className="font-mono text-sm text-muted-foreground">
                        {(pick.prob_over * 100).toFixed(0)}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <EdgeBadge edge={pick.edge} grade={pick.grade} />
                  </TableCell>
                </TableRow>
                {isExpanded && (
                  <TableRow>
                    <TableCell colSpan={7} className="bg-white/[0.01] p-0">
                      <ExpertBreakdown pick={pick} />
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
