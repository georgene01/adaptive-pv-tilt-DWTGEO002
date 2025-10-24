#!/usr/bin/env python3
"""
pvdaq1430_build_scada_daily.py
Build daily AC energy (kWh) from NREL PVDAQ #1430 CSVs.

Usage:
  python pvdaq1430_build_scada_daily.py \
    --src data_raw/validation_system_1430 \
    --out inputs/validation/scada_pvdaq1430_2017.csv

Notes:
- Prefers interval energy counter `kwh_net__5046` (treated as COUNTS).
- Auto-infers COUNTS scale in kWh per count by comparing against AC power integration
  (tries {0.01, 0.1, 1.0} kWh/count) and picks the one that best matches power-integral magnitude.
- If the counter is missing or unusable, falls back to integrating `ac_power__5074` directly.
- All daily bins are computed in America/Denver local time.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def robust_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Skip (read error): {path.name} -> {e}")
        return None

def to_denver(ts):
    t = pd.to_datetime(ts, errors="coerce", infer_datetime_format=True)
    if t.dt.tz is None:
        t = t.dt.tz_localize("America/Denver", nonexistent="shift_forward", ambiguous="NaT")
    else:
        t = t.dt.tz_convert("America/Denver")
    return t

def integrate_power_kwh(df):
    if "ac_power__5074" not in df.columns:
        return None
    t = to_denver(df["measured_on"])
    p = pd.to_numeric(df["ac_power__5074"], errors="coerce").clip(lower=0)
    s = pd.Series(p.values, index=t).sort_index()
    dt_h = (s.index.to_series().shift(-1) - s.index.to_series()).dt.total_seconds()/3600.0
    e = (s * dt_h).iloc[:-1]  # units = (unknown power units)*h
    # Decide if p is W or kW via magnitude heuristic (median power threshold 5)
    med = float(np.nanmedian(p))
    if med > 5:
        # assume kW
        daily_kwh = e.resample("1D").sum()
    else:
        # assume W
        daily_kwh = (e/1000.0).resample("1D").sum()
    daily_kwh = daily_kwh.rename("energy_kWh_power")
    return daily_kwh

def counts_kwh(df, scale_kwh_per_count):
    e = pd.to_numeric(df["kwh_net__5046"], errors="coerce")
    de = e.diff()
    de = de.where(de >= 0, 0.0)  # ignore resets
    t = to_denver(df["measured_on"])
    s = pd.Series(de.values, index=t)
    daily_kwh = (s.resample("1D").sum() * scale_kwh_per_count).rename("energy_kWh_counts")
    return daily_kwh

def best_scale_vs_power(df):
    # Try COUNTS scales and choose the one nearest to power-integral where both exist
    cand_scales = [0.01, 0.1, 1.0]   # kWh per count
    d_power = integrate_power_kwh(df)
    if d_power is None or d_power.empty:
        return None  # no reference
    errs = []
    for s in cand_scales:
        d_counts = counts_kwh(df, s) if "kwh_net__5046" in df.columns else None
        if d_counts is None or d_counts.empty:
            errs.append((np.inf, s))
            continue
        # compare only overlapping days
        join = d_power.to_frame().join(d_counts, how="inner")
        if join.empty:
            errs.append((np.inf, s))
            continue
        # use relative error of totals as score
        tot_p = join["energy_kWh_power"].sum()
        tot_c = join["energy_kWh_counts"].sum()
        if tot_p <= 0 or tot_c <= 0:
            errs.append((np.inf, s))
            continue
        rel = abs(tot_p - tot_c)/tot_p
        errs.append((rel, s))
    errs.sort(key=lambda x: x[0])
    return errs[0][1] if errs and np.isfinite(errs[0][0]) else None

def daily_energy_from_file(path, forced_scale=None):
    df = robust_read_csv(path)
    if df is None or "measured_on" not in df.columns:
        return None
    # prefer counts if present
    scale = forced_scale
    if scale is None and "kwh_net__5046" in df.columns:
        scale = best_scale_vs_power(df)
    if scale is None and "kwh_net__5046" in df.columns:
        # fallback to 0.1 kWh/count if no power reference available
        scale = 0.1
    out = []
    if "kwh_net__5046" in df.columns and scale is not None:
        d_counts = counts_kwh(df, scale)
        out.append(d_counts)
    d_power = integrate_power_kwh(df)
    if d_power is not None:
        out.append(d_power)
    if not out:
        return None
    # Prefer counts where available; else power
    d = pd.concat(out, axis=1)
    if "energy_kWh_counts" in d.columns:
        d["energy_kWh"] = d["energy_kWh_counts"].fillna(d["energy_kWh_power"])
    else:
        d["energy_kWh"] = d["energy_kWh_power"]
    return d["energy_kWh"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with PVDAQ CSVs (system_1430__date_*.csv)")
    ap.add_argument("--out", required=True, help="Output CSV path for daily measured kWh")
    ap.add_argument("--scale", type=float, default=None, help="Force kWh per count for kwh_net__5046 (overrides auto)")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(src.glob("system_1430__date_*.csv"))
    rows = []
    for f in files:
        series = daily_energy_from_file(f, forced_scale=args.scale)
        if series is None or series.empty: 
            print(f"Skip (no usable data): {f.name}")
            continue
        for ts, val in series.items():
            rows.append({"site":"PVDAQ1430","date_local":ts.strftime("%Y-%m-%d"),"energy_kWh":float(val)})

    out = (pd.DataFrame(rows)
           .drop_duplicates(subset=["date_local"])
           .sort_values("date_local")
           .reset_index(drop=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")
    if len(out):
        print(out.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
