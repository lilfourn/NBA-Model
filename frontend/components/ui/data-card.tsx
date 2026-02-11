"use client";

import { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface DataCardProps {
  title: ReactNode;
  description?: ReactNode;
  loading?: boolean;
  noData?: boolean;
  noDataDescription?: ReactNode;
  error?: ReactNode;
  errorDescription?: ReactNode;
  children?: ReactNode;
  skeletonHeight?: string;
}

export function DataCard({
  title,
  description,
  loading = false,
  noData = false,
  noDataDescription,
  error,
  errorDescription,
  children,
  skeletonHeight = "h-72",
}: DataCardProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <Skeleton className={`w-full ${skeletonHeight}`} />
        </CardContent>
      </Card>
    );
  }

  if (noData && !children) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>
            {noDataDescription ?? "No data available yet."}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        {error && (
          <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-500">
            <p className="font-medium">{error}</p>
            {errorDescription && (
              <p className="mt-1 text-xs text-amber-500/90">{errorDescription}</p>
            )}
          </div>
        )}
        {noData ? (
          <p className="text-sm text-muted-foreground">
            {noDataDescription ?? "No data available yet."}
          </p>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}
