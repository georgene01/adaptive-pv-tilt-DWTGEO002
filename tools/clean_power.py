#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, pytz, os, sys
tz = pytz.timezone("Africa/Johannesburg")
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
args = ap.parse_args()
if not os.path.exists(args.inp): sys.exit(f"[ERROR] not found: {args.inp}")
df = pd.read_csv(args.inp)
if "time_utc" not in df.columns: sys.exit("[ERROR] 'time_utc' missing")
df["time_utc"] = pd.to_datetime(df["time_utc"].astype(str).str.strip(), errors="coerce", utc=True)
df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
df["time_local"] = df["time_utc"].dt.tz_convert(tz)
df["local_date"] = df["time_local"].dt.strftime("%Y-%m-%d")
df["local_hour"] = df["time_local"].dt.hour
ghi = None
for c in ["ghi_wm2","GHI","ghi","GHI_wm2"]:
    if c in df.columns: ghi = c; break
if ghi is not None:
    x = pd.to_numeric(df[ghi], errors="coerce")
    q95 = np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan
    if np.isfinite(q95) and q95 <= 5: df[ghi] = x*1000.0
out = os.path.splitext(args.inp)[0] + "_LOCAL.csv"
df.to_csv(out, index=False)
df.groupby("local_date").size().rename("rows").reset_index().to_csv(os.path.splitext(args.inp)[0]+"_LOCAL_day_counts.csv", index=False)
print(f"[OK] wrote: {out}")
