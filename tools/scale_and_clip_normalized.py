#!/usr/bin/env python3
import argparse, sys, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

CANDIDATE_PERKWDC = [
    "ac_per_kwdc", "pac_per_kwdc", "p_ac_per_kwdc", "ac_kw_per_kwdc",
    "P_ac_per_kwdc", "P_ac_norm_kw_per_kwdc", "P_ac_norm"
]
CANDIDATE_TIME = ["time_local","time_utc","timestamp","time","DateTime"]

def infer_dt_minutes(df, time_cols):
    if time_cols:
        tcol = time_cols[0]
        t = pd.to_datetime(df[tcol])
        if len(t) >= 2:
            dt = np.median(np.diff(t.values).astype('timedelta64[m]').astype(float))
            if np.isfinite(dt) and dt > 0:
                return float(dt)
    # fallback: try to detect by length (hourly or 15-min typical)
    n = len(df)
    for guess in (60, 30, 15):
        # rough: if a full clear day ~ 24h, expect ~ 24*60/guess rows
        expected = 24*60/guess
        if 0.5*expected <= n <= 2.0*expected:
            return float(guess)
    return 60.0

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--policy", required=True, choices=["baseline","offpoint","bin","annual"])
    ap.add_argument("--input_glob", required=True, help="Files of normalized hourlies (CSV/Parquet)")
    ap.add_argument("--per_kwdc_col", default=None, help="Override: AC-per-kWdc column name")
    ap.add_argument("--ac_kw_col", default=None, help="If data are already plant-scale AC kW")
    ap.add_argument("--pac_kw", type=float, required=True, help="Plant AC nameplate (kW)")
    ap.add_argument("--dcac", type=float, required=True, help="DC/AC ratio (e.g., 1.30)")
    ap.add_argument("--dt_minutes", type=float, default=None, help="Override step minutes")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print(f"ERROR: No files matched input_glob={args.input_glob}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out_csv)
    write_header = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plant_dc_kw = args.pac_kw * args.dcac

    totals_clip = 0.0
    totals_ac   = 0.0
    rows = []

    for fp in files:
        # read CSV or Parquet
        try:
            if fp.lower().endswith(".parquet"):
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)
        except Exception as e:
            print(f"WARN: skip {fp}: {e}", file=sys.stderr); continue

        # choose AC column
        per_kwdc_col = args.per_kwdc_col or find_col(df, CANDIDATE_PERKWDC)
        ac_kw_col    = args.ac_kw_col

        if per_kwdc_col is None and ac_kw_col is None:
            # Try sensible fallbacks often seen in your pipeline
            if "P_ac_raw_kw" in df.columns and "P_ac_clip_kw" not in df.columns:
                # Likely per-kWdc scaled to 1 kWdc? Heuristic: if max<5, treat as per-kWdc
                if float(pd.to_numeric(df["P_ac_raw_kw"], errors="coerce").fillna(0).max()) < 5.0:
                    per_kwdc_col = "P_ac_raw_kw"
                else:
                    ac_kw_col = "P_ac_raw_kw"

        if per_kwdc_col is None and ac_kw_col is None:
            print(f"ERROR: {fp} has no recognizable AC-per-kWdc or AC-kW column. "
                  f"Tried {CANDIDATE_PERKWDC} and 'P_ac_raw_kw'. Use --per_kwdc_col/--ac_kw_col.",
                  file=sys.stderr)
            continue

        # interval minutes
        time_cols = [c for c in CANDIDATE_TIME if c in df.columns]
        dt_min = args.dt_minutes or infer_dt_minutes(df, time_cols)

        if per_kwdc_col:
            ac_raw_kw = pd.to_numeric(df[per_kwdc_col], errors="coerce").fillna(0.0) * plant_dc_kw
        else:
            ac_raw_kw = pd.to_numeric(df[ac_kw_col], errors="coerce").fillna(0.0)

        ac_clip_kw = np.minimum(ac_raw_kw.values, args.pac_kw)
        clip_kw    = np.maximum(ac_raw_kw.values - args.pac_kw, 0.0)

        e_ac_kwh   = ac_clip_kw.sum() * (dt_min/60.0)
        e_clip_kwh = clip_kw.sum()    * (dt_min/60.0)

        # infer date from filename or data
        date_str = None
        for cand in ("date_local","date","Date","day","Day","DATE"):
            if cand in df.columns:
                v = str(df[cand].iloc[0])
                if v and v.lower() != "nan":
                    date_str = v.split()[0]; break
        if date_str is None:
            # pull a YYYY-MM-DD from filename if present
            import re
            m = re.search(r"(20\d{2}-\d{2}-\d{2})", fp)
            date_str = m.group(1) if m else "YYYY-MM-DD"

        rows.append((date_str, args.site, args.policy, e_clip_kwh, e_ac_kwh))
        totals_clip += e_clip_kwh
        totals_ac   += e_ac_kwh

    # write/append daily rows
    if rows:
        df_out = pd.DataFrame(rows, columns=["date","site","policy","clip_kWh_day","ac_kWh_day"])
        df_out.sort_values(["date"], inplace=True)
        df_out.to_csv(out_path, mode="a", header=write_header, index=False)

    print(json.dumps({
        "files_processed": len(rows),
        "site": args.site,
        "policy": args.policy,
        "PAC_kW": args.pac_kw,
        "DCAC": args.dcac,
        "annual_clip_MWh": totals_clip/1000.0,
        "annual_ac_MWh": totals_ac/1000.0,
        "out_csv": str(out_path)
    }))
if __name__ == "__main__":
    main()
