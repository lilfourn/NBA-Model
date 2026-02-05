"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface StatTypeFilterProps {
  statTypes: string[];
  value: string;
  onChange: (value: string) => void;
}

export function StatTypeFilter({
  statTypes,
  value,
  onChange,
}: StatTypeFilterProps) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="All Stats" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">All Stats</SelectItem>
        {statTypes.map((st) => (
          <SelectItem key={st} value={st}>
            {st}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
