from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Orchestrate Weekly Notes analysis + aggregation")
    p.add_argument("--cleaned-json", default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()))
    p.add_argument("--hours", default="24,72,168", help="Comma-separated horizons")
    p.add_argument("--limit", type=int, default=0, help="Limit articles (0=all)")
    p.add_argument("--persist-db", action="store_true", help="Upsert results into Postgres if DATABASE_URL set")
    p.add_argument("--out-csv", default=str(Path("test_results/weekly_notes_analysis.csv").resolve()))
    p.add_argument("--summary-json", default=str(Path("test_results/weekly_notes_summary.json").resolve()))
    p.add_argument("--summary-csv", default=str(Path("test_results/weekly_notes_summary.csv").resolve()))
    p.add_argument("--horizon-for-summary", type=int, default=168)
    return p


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    args = build_argparser().parse_args()

    analyzer_cmd = [
        sys.executable, "-m", "app.analytics.weekly_notes_analyzer",
        "--cleaned-json", args.cleaned_json,
        "--output", args.out_csv,
        "--hours", args.hours,
    ]
    if args.limit and args.limit > 0:
        analyzer_cmd += ["--limit", str(args.limit)]
    if args.persist_db:
        analyzer_cmd += ["--persist-db"]

    rc = run(analyzer_cmd)
    if rc != 0:
        return rc

    aggr_cmd = [
        sys.executable, "-m", "app.analytics.weekly_notes_aggregations",
        "--input", args.out_csv,
        "--horizon", str(args.horizon_for_summary),
        "--output-json", args.summary_json,
        "--output-csv", args.summary_csv,
    ]
    rc = run(aggr_cmd)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
