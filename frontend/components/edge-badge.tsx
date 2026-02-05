import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const GRADE_COLORS: Record<string, string> = {
  "A+": "bg-amber-500/15 text-amber-400 border-amber-500/20",
  A: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20",
  B: "bg-sky-500/15 text-sky-400 border-sky-500/20",
  C: "bg-zinc-500/15 text-zinc-400 border-zinc-500/20",
  D: "bg-orange-500/15 text-orange-400 border-orange-500/20",
  F: "bg-red-500/15 text-red-400 border-red-500/20",
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
