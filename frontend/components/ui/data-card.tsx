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
  children?: ReactNode;
  skeletonHeight?: string;
}

export function DataCard({
  title,
  description,
  loading = false,
  noData = false,
  noDataDescription,
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

  if (noData) {
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
      {children && <CardContent>{children}</CardContent>}
    </Card>
  );
}
