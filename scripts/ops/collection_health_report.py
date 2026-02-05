"""Generate a collection health report from structured logs.

Usage:
    python -m scripts.ops.collection_health_report [--log-path logs/collection.jsonl] [--hours 24]
"""
from __future__ import annotations

import argparse
import json
import sys

from app.clients.logging import generate_health_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collection health report")
    parser.add_argument("--log-path", default="logs/collection.jsonl")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    report = generate_health_report(args.log_path, hours=args.hours)
    print(json.dumps(report, indent=2))

    if "error" in report:
        sys.exit(1)

    for source, stats in report.get("sources", {}).items():
        if stats.get("error_rate", 0) > 0.3:
            print(f"WARNING: {source} error rate {stats['error_rate']:.1%}", file=sys.stderr)
        if stats.get("circuit_opens", 0) > 0:
            print(f"WARNING: {source} had {stats['circuit_opens']} circuit breaker opens", file=sys.stderr)


if __name__ == "__main__":
    main()
