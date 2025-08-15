from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AggregationConfig:
    horizon_hours: int = 168  # default 1w
    assets: List[str] = None

    def __post_init__(self):
        if self.assets is None:
            self.assets = ["BTC", "ETH"]


def _load_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    import csv
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def compute_metrics(rows: List[Dict[str, Any]], cfg: AggregationConfig) -> Dict[str, Any]:
    if not rows:
        return {"assets": {}}

    df = pd.DataFrame(rows)
    # Coerce types
    if "published_at_utc" in df.columns:
        df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], errors="coerce")
        df["week"] = df["published_at_utc"].dt.to_period("W")
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce").fillna(0.0)

    # Normalize sentiment to three buckets
    def norm_sent(x: Any) -> str:
        s = str(x or "").lower()
        if "bull" in s:
            return "bullish"
        if "bear" in s:
            return "bearish"
        return "neutral"

    df["sentiment_bucket"] = df.get("sentiment", "neutral").map(norm_sent)

    summary: Dict[str, Any] = {"assets": {}, "horizon_hours": cfg.horizon_hours}

    for asset in cfg.assets:
        ret_col = f"{asset}_ret_{cfg.horizon_hours}h"
        if ret_col not in df.columns:
            continue
        sdf = df[["sentiment_bucket", ret_col, "week"]].copy()
        sdf[ret_col] = pd.to_numeric(sdf[ret_col], errors="coerce")

        # Bucketed returns stats
        bs_df = (
            sdf.groupby("sentiment_bucket", observed=True)[ret_col]
            .agg(["count", "mean", "median"])
            .rename(columns={"mean": "avg_return", "median": "median_return"})
            .reset_index()
        )
        # Ensure numeric dtypes where applicable before rounding
        for col in ["avg_return", "median_return"]:
            if col in bs_df.columns:
                bs_df[col] = pd.to_numeric(bs_df[col], errors="coerce")
                bs_df[col] = bs_df[col].round(6)
        bucket_stats = bs_df.to_dict(orient="records")

        # Hit-rate: did move align with sentiment direction?
        def direction_hit(row) -> Optional[int]:
            r = row.get(ret_col)
            s = row.get("sentiment_bucket")
            if pd.isna(r):
                return None
            if s == "bullish":
                return int(r > 0)
            if s == "bearish":
                return int(r < 0)
            # neutral: treat as NA
            return None

        hits = sdf.copy()
        hits["hit"] = hits.apply(direction_hit, axis=1)
        hr_df = (
            hits.dropna(subset=["hit"])
            .groupby("sentiment_bucket", observed=True)["hit"]
            .mean()
            .reset_index()
            .rename(columns={"hit": "hit_rate"})
        )
        if "hit_rate" in hr_df.columns:
            hr_df["hit_rate"] = pd.to_numeric(hr_df["hit_rate"], errors="coerce").round(6)
        hit_rate = hr_df.to_dict(orient="records")

        # Rolling 4-week average returns by bucket
        roll = sdf.copy()
        roll = roll.dropna(subset=[ret_col])
        roll = roll.sort_values(by="week")
        # weekly means then per-bucket rolling transform
        weekly = (
            roll.groupby(["sentiment_bucket", "week"], observed=True)[ret_col]
            .mean()
            .rename("weekly_mean")
            .reset_index()
        )
        weekly["rolling_avg_return"] = (
            weekly.sort_values(["sentiment_bucket", "week"])  # ensure order within bucket
            .groupby("sentiment_bucket", observed=True)["weekly_mean"]
            .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        )
        rolling_trends = weekly[["sentiment_bucket", "week", "rolling_avg_return"]].copy()
        # Convert Period to string for JSON/CSV
        if "week" in rolling_trends.columns:
            rolling_trends["week"] = rolling_trends["week"].astype(str)
        rolling_records = rolling_trends.to_dict(orient="records")

        summary["assets"][asset] = {
            "bucket_stats": bucket_stats,
            "hit_rates": hit_rate,
            "rolling_4w_trends": rolling_records,
            "sample_size": int(len(sdf) - int(sdf[ret_col].isna().sum())),
        }

    return summary


def main():
    ap = argparse.ArgumentParser(description="Aggregate Weekly Notes metrics from per-article CSV")
    ap.add_argument("--input", default=str(Path("test_results/weekly_notes_analysis.csv").resolve()))
    ap.add_argument("--horizon", default=168, type=int)
    ap.add_argument("--output-json", default=str(Path("test_results/weekly_notes_summary.json").resolve()))
    ap.add_argument("--output-csv", default=str(Path("test_results/weekly_notes_summary.csv").resolve()))
    args = ap.parse_args()

    rows = _load_rows_from_csv(Path(args.input))
    cfg = AggregationConfig(horizon_hours=args.horizon)
    summary = compute_metrics(rows, cfg)

    # Write JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    # Flatten for CSV: bucket_stats per asset
    flat_rows: List[Dict[str, Any]] = []
    for asset, sec in summary.get("assets", {}).items():
        for rec in sec.get("bucket_stats", []):
            flat = {"asset": asset, **rec}
            flat_rows.append(flat)
    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(args.output_csv, index=False)
    print(f"Wrote summary to {out_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
