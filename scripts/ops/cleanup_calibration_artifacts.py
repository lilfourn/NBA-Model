import argparse
from pathlib import Path


def _sorted_by_mtime(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _prune(*, paths: list[Path], keep: int) -> tuple[list[Path], list[Path]]:
    if keep < 0:
        keep = 0
    ordered = _sorted_by_mtime(paths)
    return ordered[:keep], ordered[keep:]


def _collect(dir_path: Path, pattern: str) -> list[Path]:
    return [
        p
        for p in dir_path.glob(pattern)
        if p.is_file() and p.name != ".gitkeep"
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete older calibration/backtest artifacts under data/calibration/.")
    ap.add_argument("--dir", default="data/calibration")
    ap.add_argument("--keep-calibrations", type=int, default=3)
    ap.add_argument("--keep-reports", type=int, default=3)
    ap.add_argument("--keep-backtests", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="Actually delete files (default: dry-run).")
    args = ap.parse_args()

    base = Path(args.dir)
    reports = base / "reports"

    calib_files = _collect(base, "forecast_calibration*.json")
    report_files = _collect(reports, "calibration_report*.csv")
    backtest_files = _collect(base, "forecast_backtest*.csv")

    keep_calib, del_calib = _prune(paths=calib_files, keep=int(args.keep_calibrations))
    keep_reports, del_reports = _prune(paths=report_files, keep=int(args.keep_reports))
    keep_backtests, del_backtests = _prune(paths=backtest_files, keep=int(args.keep_backtests))

    def fmt(paths: list[Path]) -> list[str]:
        return [str(p) for p in paths]

    print(
        {
            "found": {
                "calibrations": len(calib_files),
                "reports": len(report_files),
                "backtests": len(backtest_files),
            },
            "keep": {
                "calibrations": fmt(keep_calib),
                "reports": fmt(keep_reports),
                "backtests": fmt(keep_backtests),
            },
            "delete": {
                "calibrations": fmt(del_calib),
                "reports": fmt(del_reports),
                "backtests": fmt(del_backtests),
            },
            "execute": bool(args.execute),
        }
    )

    if not args.execute:
        print("Dry-run only. Re-run with --execute to delete files.")
        return

    deleted = 0
    for p in [*del_calib, *del_reports, *del_backtests]:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    print({"deleted_files": deleted})


if __name__ == "__main__":
    main()

