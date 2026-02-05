import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function ConfidenceBadge({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  const color =
    value >= 0.6
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
      : value >= 0.55
        ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
        : "bg-muted text-muted-foreground";

  return (
    <Badge variant="outline" className={cn("font-mono text-xs", color)}>
      {pct}%
    </Badge>
  );
}
