#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize which bin is most common by month and by season (per site).

Inputs (already in your repo):
  bins/bins_NC_2024.csv
  bins/bins_WC_2024.csv
(They must contain at least: 'date' and 'bin_label')

Outputs:
  tables/bin_mode_by_month_{SITE}.csv
  tables/bin_mode_by_season_{SITE}.csv
  tables/bin_counts_by_month_{SITE}.csv           # full distribution per month
  tables/bin_counts_by_season_{SITE}.csv          # full distribution per season
And prints compact human-readable summaries for both sites.
"""

import os
import pandas as pd
from datetime import datetime

SITES = ["NC", "WC"]

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_bins(site: str) -> pd.DataFrame:
    f = f"bins/bins_{site}_2024.csv"
    df = pd.read_csv(f)
    if "date" not in df.columns or "bin_label" not in df.columns:
        raise SystemExit(f"[FATAL] {f} must contain 'date' and 'bin_label'; got {list(df.columns)}")
    # normalize date → YYYY-MM-DD
    d = df["date"].astype(str).str.strip().str.replace(r"[\\/\.]", "-", regex=True)
    # handle YYYYMMDD
    mask8 = d.str.fullmatch(r"\d{8}", na=False)
    d.loc[mask8] = d.loc[mask8].str.replace(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True)
    df["date"] = pd.to_datetime(d, errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    # Seasons for Southern Hemisphere (South Africa): DJF, MAM, JJA, SON
    def season_row(m):
        if m in (12,1,2):  return "DJF (summer)"
        if m in (3,4,5):   return "MAM (autumn)"
        if m in (6,7,8):   return "JJA (winter)"
        return "SON (spring)"  # 9,10,11
    df["season"] = df["date"].dt.month.map(season_row)
    df["bin_id"] = df["bin_label"].astype(str)
    return df

def mode_table(grouped_counts: pd.DataFrame, total_key: str) -> pd.DataFrame:
    """
    grouped_counts: index = (period, bin_id) with column 'n'
    total_key: 'month' or 'season'
    returns table with columns: [period, top_bin, top_count, pct, total_days]
    """
    counts = grouped_counts.reset_index()
    totals = counts.groupby(total_key)["n"].sum().rename("total_days")
    idxmax = counts.groupby(total_key).apply(lambda x: x.loc[x["n"].idxmax()])
    idxmax = idxmax.reset_index(drop=True).rename(columns={"bin_id":"top_bin", "n":"top_count"})
    out = idxmax.merge(totals, on=total_key, how="left")
    out["pct"] = 100.0 * out["top_count"] / out["total_days"]
    return out

def run_for_site(site: str):
    df = load_bins(site)

    # ---- By month ----
    by_month = (df.groupby(["month","month_name","bin_id"])
                  .size().rename("n").reset_index())
    # save full distribution by month (pivot)
    month_pivot = by_month.pivot_table(index=["month","month_name"], columns="bin_id", values="n", fill_value=0).sort_index()
    ensure_dir(f"tables/bin_counts_by_month_{site}.csv")
    month_pivot.to_csv(f"tables/bin_counts_by_month_{site}.csv")
    # mode per month
    month_counts = by_month.set_index(["month","month_name","bin_id"])["n"]
    month_mode = mode_table(month_counts.reset_index().set_index(["month","bin_id"])["n"].rename_axis(["month","bin_id"]).reset_index(), "month")
    # bring back month_name for readability
    month_mode = month_mode.merge(df[["month","month_name"]].drop_duplicates(), on="month", how="left") \
                           .drop_duplicates(subset=["month"]) \
                           .sort_values("month")
    ensure_dir(f"tables/bin_mode_by_month_{site}.csv")
    month_mode[["month","month_name","top_bin","top_count","total_days","pct"]].to_csv(f"tables/bin_mode_by_month_{site}.csv", index=False)

    # ---- By season ----
    by_season = (df.groupby(["season","bin_id"])
                   .size().rename("n").reset_index())
    season_pivot = by_season.pivot_table(index="season", columns="bin_id", values="n", fill_value=0).reindex(["DJF (summer)","MAM (autumn)","JJA (winter)","SON (spring)"])
    ensure_dir(f"tables/bin_counts_by_season_{site}.csv")
    season_pivot.to_csv(f"tables/bin_counts_by_season_{site}.csv")
    season_counts = by_season.set_index(["season","bin_id"])["n"].reset_index()
    season_mode = mode_table(season_counts.set_index(["season","bin_id"])["n"].rename_axis(["season","bin_id"]).reset_index(), "season")
    season_mode = season_mode.set_index("season").loc(["DJF (summer)","MAM (autumn)","JJA (winter)","SON (spring)"]) if isinstance(season_mode.index, pd.MultiIndex) else season_mode
    ensure_dir(f"tables/bin_mode_by_season_{site}.csv")
    season_mode[["season","top_bin","top_count","total_days","pct"]].to_csv(f"tables/bin_mode_by_season_{site}.csv", index=False)

    # ---- Print summaries ----
    print(f"\n=== {site}: Most common bin by MONTH ===")
    print(month_mode[["month","month_name","top_bin","top_count","total_days","pct"]].to_string(index=False, formatters={"pct":lambda x: f"{x:.1f}%"}))
    print(f"\n[OK] wrote tables/bin_mode_by_month_{site}.csv and tables/bin_counts_by_month_{site}.csv")

    print(f"\n=== {site}: Most common bin by SEASON ===")
    season_mode_ordered = season_mode.set_index("season").reindex(["DJF (summer)","MAM (autumn)","JJA (winter)","SON (spring)"]).reset_index()
    print(season_mode_ordered[["season","top_bin","top_count","total_days","pct"]].to_string(index=False, formatters={"pct":lambda x: f"{x:.1f}%"}))
    print(f"\n[OK] wrote tables/bin_mode_by_season_{site}.csv and tables/bin_counts_by_season_{site}.csv")

def main():
    for site in SITES:
        run_for_site(site)

if __name__ == "__main__":
    main()
