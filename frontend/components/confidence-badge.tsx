import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function ConfidenceBadge({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  const color =
    value >= 0.6
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/20"
      : value >= 0.55
        ? "bg-amber-500/15 text-amber-400 border-amber-500/20"
        : "bg-white/[0.04] text-muted-foreground border-border";

  return (
    <Badge variant="outline" className={cn("font-mono text-xs", color)}>
      {pct}%
    </Badge>
  );
}
