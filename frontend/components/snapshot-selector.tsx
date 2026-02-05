"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Snapshot } from "@/lib/api";

interface SnapshotSelectorProps {
  snapshots: Snapshot[];
  value: string;
  onChange: (id: string) => void;
}

function formatDate(iso: string | null): string {
  if (!iso) return "Unknown";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function SnapshotSelector({
  snapshots,
  value,
  onChange,
}: SnapshotSelectorProps) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-[260px]">
        <SelectValue placeholder="Select snapshot" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="latest">Latest Snapshot</SelectItem>
        {snapshots.map((s) => (
          <SelectItem key={s.id} value={s.id}>
            {formatDate(s.fetched_at)}
            {s.included_count !== null && (
              <span className="text-muted-foreground ml-2">
                ({s.included_count} projections)
              </span>
            )}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
