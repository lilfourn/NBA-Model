"use client";

import { TrainingHistoryChart } from "@/components/stats/training-history-chart";
import { ExpertComparisonChart } from "@/components/stats/expert-comparison-chart";
import { HitRateChart } from "@/components/stats/hit-rate-chart";
import { CalibrationChart } from "@/components/stats/calibration-chart";
import { EnsembleWeightsChart } from "@/components/stats/ensemble-weights-chart";
import { ConfidenceDistChart } from "@/components/stats/confidence-dist-chart";

export default function StatsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h1 className="text-xl font-semibold tracking-tight">Stats</h1>
          <p className="text-sm text-muted-foreground">
            Training performance, prediction accuracy, and ensemble diagnostics
          </p>
        </div>

        <div className="space-y-6">
          <ExpertComparisonChart />
          <TrainingHistoryChart />
          <HitRateChart />
          <EnsembleWeightsChart />
          <div className="grid gap-6 lg:grid-cols-2">
            <CalibrationChart />
            <ConfidenceDistChart />
          </div>
        </div>
      </div>
    </div>
  );
}
