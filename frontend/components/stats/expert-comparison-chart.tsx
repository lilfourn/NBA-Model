"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchExpertComparison } from "@/lib/api";
import type { ExpertSummary } from "@/lib/api";

const MODEL_COLORS: Record<string, string> = {
  baseline_logreg: "var(--chart-1)",
  nn_gru_attention: "var(--chart-2)",
  xgboost: "var(--chart-3)",
  lightgbm: "var(--chart-4)",
};

const MODEL_LABELS: Record<string, string> = {
  baseline_logreg: "LR",
  nn_gru_attention: "NN",
  xgboost: "XGB",
  lightgbm: "LGBM",
};

export function ExpertComparisonChart() {
  const [experts, setExperts] = useState<ExpertSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExpertComparison()
      .then((r) => setExperts(r.experts))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Expert Comparison</CardTitle></CardHeader>
        <CardContent><Skeleton className="h-72 w-full" /></CardContent>
      </Card>
    );
  }

  if (experts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Expert Comparison</CardTitle>
          <CardDescription>No model runs available yet.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const accData = experts.map((e) => ({
    name: MODEL_LABELS[e.model_name] || e.model_name,
    value: e.accuracy ? +(e.accuracy * 100).toFixed(1) : 0,
    color: MODEL_COLORS[e.model_name] || "var(--chart-4)",
  }));

  const aucData = experts.map((e) => ({
    name: MODEL_LABELS[e.model_name] || e.model_name,
    value: e.roc_auc ? +e.roc_auc.toFixed(4) : 0,
    color: MODEL_COLORS[e.model_name] || "var(--chart-4)",
  }));

  const conformalData = experts.map((e) => ({
    name: MODEL_LABELS[e.model_name] || e.model_name,
    value: e.conformal_q_hat ? +e.conformal_q_hat.toFixed(4) : 0,
    color: MODEL_COLORS[e.model_name] || "var(--chart-4)",
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Expert Comparison</CardTitle>
        <CardDescription>Latest metrics for each model expert</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Summary stat cards */}
        <div className="mb-6 grid grid-cols-3 gap-3">
          {experts.map((e) => (
            <div
              key={e.model_name}
              className="rounded-lg border p-3 text-center"
            >
              <p className="text-xs text-muted-foreground">{MODEL_LABELS[e.model_name] || e.model_name}</p>
              <p className="text-2xl font-bold tabular-nums">
                {e.accuracy ? `${(e.accuracy * 100).toFixed(1)}%` : "—"}
              </p>
              <p className="text-xs text-muted-foreground">
                AUC {e.roc_auc?.toFixed(3) ?? "—"} · {e.train_rows.toLocaleString()} rows
              </p>
            </div>
          ))}
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div>
            <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">Accuracy %</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={accData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {accData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div>
            <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">ROC AUC</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={aucData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.5, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {aucData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div>
            <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">Conformal q̂</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={conformalData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {conformalData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
