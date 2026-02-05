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
          <h1 className="text-2xl font-bold tracking-tight">Model Statistics</h1>
          <p className="text-sm text-muted-foreground">
            Training performance, prediction accuracy, and ensemble diagnostics
          </p>
        </div>

        <div className="space-y-6">
          {/* Row 1: Expert comparison + Training history */}
          <div className="grid gap-6 lg:grid-cols-2">
            <ExpertComparisonChart />
            <TrainingHistoryChart />
          </div>

          {/* Row 2: Hit rate (full width) */}
          <HitRateChart />

          {/* Row 3: Ensemble weights (full width) */}
          <EnsembleWeightsChart />

          {/* Row 4: Calibration + Confidence dist */}
          <div className="grid gap-6 lg:grid-cols-2">
            <CalibrationChart />
            <ConfidenceDistChart />
          </div>
        </div>
      </div>
    </div>
  );
}
