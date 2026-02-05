import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const GRADE_COLORS: Record<string, string> = {
  "A+": "bg-emerald-200 text-emerald-900 dark:bg-emerald-800 dark:text-emerald-100",
  A: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  B: "bg-sky-100 text-sky-800 dark:bg-sky-900 dark:text-sky-200",
  C: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  D: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  F: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

export function EdgeBadge({ edge, grade }: { edge: number; grade: string }) {
  const color = GRADE_COLORS[grade] || GRADE_COLORS.F;
  return (
    <Badge variant="outline" className={cn("font-mono text-xs gap-1", color)}>
      <span className="font-semibold">{grade}</span>
      <span className="opacity-70">{Math.round(edge)}</span>
    </Badge>
  );
}
