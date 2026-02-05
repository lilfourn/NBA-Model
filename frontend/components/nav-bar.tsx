"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Trophy } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Picks", icon: Trophy },
  { href: "/stats", label: "Model Stats", icon: BarChart3 },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-card">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6 lg:px-8">
        <span className="text-sm font-bold tracking-tight">NBA Picks</span>
        <div className="flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                }`}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
