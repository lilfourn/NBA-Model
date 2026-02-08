from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

FAMILY_PATTERNS = {
    "baseline_logreg": "baseline_logreg_*.joblib",
    "xgb": "xgb_*.joblib",
    "lgbm": "lgbm_*.joblib",
    "meta_learner": "meta_learner_*.joblib",
    "nn_gru_attention": "nn_gru_attention_*.pt",
    "tabdl": "tabdl_*.pt",
}


def _sorted_candidates(models_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        models_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Archive or delete stale local model artifacts per family."
    )
    ap.add_argument("--models-dir", default="models")
    ap.add_argument(
        "--keep",
        type=int,
        default=2,
        help="How many newest artifacts to keep per model family.",
    )
    ap.add_argument(
        "--archive-dir",
        default="",
        help="Optional archive directory. Defaults to models/archive/<timestamp>.",
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Delete stale artifacts instead of moving to archive.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep = max(1, int(args.keep))
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    archive_dir = (
        Path(args.archive_dir)
        if args.archive_dir
        else (models_dir / "archive" / f"pruned_{timestamp}")
    )

    total_stale = 0
    moved_or_deleted = 0
    for family, pattern in FAMILY_PATTERNS.items():
        candidates = _sorted_candidates(models_dir, pattern)
        stale = candidates[keep:]
        total_stale += len(stale)
        print(
            f"{family}: total={len(candidates)} keep={min(len(candidates), keep)} stale={len(stale)}"
        )
        if not stale:
            continue

        if not args.delete and not args.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)

        for src in stale:
            if args.dry_run:
                print(f"  DRY-RUN: {'delete' if args.delete else 'archive'} {src.name}")
                continue
            if args.delete:
                src.unlink(missing_ok=True)
            else:
                dest = archive_dir / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dest)
            moved_or_deleted += 1

    if args.dry_run:
        print(f"Dry run complete. Total stale files: {total_stale}")
        return

    if args.delete:
        print(f"Deleted {moved_or_deleted} stale artifacts.")
    else:
        if moved_or_deleted:
            print(f"Archived {moved_or_deleted} stale artifacts -> {archive_dir}")
        else:
            print("No stale artifacts found.")


if __name__ == "__main__":
    main()
