from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path

# Columns we might see
TIME_CANDS = ["timestamp","time","datetime","localtime","utc_time","date_time","DateTime","Date"]
PWR_CANDS  = ["ac_power","ac_power_w","ac_kw","power_ac_w","p_ac","kw_ac","power_kw","ac power","kw"]
# Already-daily energy variants
ENERGY_DAILY_CANDS = [
    "ac_energy_daily_sum", "daily_ac_energy", "energy_kwh", "energy_mwh",
    "AC Energy (kWh)", "Daily Yield (kWh)", "daily_energy_kwh"
]

def pick(colnames, candidates):
    low = {c.lower(): c for c in colnames}
    for k in candidates:
        if k.lower() in low: return low[k.lower()]
    return None

def to_daily_from_power(df: pd.DataFrame, tz="Africa/Johannesburg") -> pd.DataFrame | None:
    tcol = pick(df.columns, TIME_CANDS)
    pcol = pick(df.columns, PWR_CANDS)
    if not tcol or not pcol:
        return None
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    p  = pd.to_numeric(df[pcol], errors="coerce")
    good = ts.notna() & p.notna()
    if not good.any():
        return None
    s = pd.Series(p[good].values, index=ts[good]).sort_index()
    # Heuristic units: if median<50 → kW, else W
    sW = s*1000.0 if np.nanmedian(s.values) < 50 else s
    # Regularize to 15-min then integrate
    sW = sW.resample("15min").interpolate(limit_direction="both")
    eWh = sW * 0.25  # Wh per 15-min
    eWh.index = eWh.index.tz_convert(tz)
    daily = (eWh.resample("1D").sum()/1000.0).rename("energy_kWh").reset_index()
    daily.rename(columns={"index":"date_local", "time":"date_local"}, inplace=True)
    daily["date_local"] = pd.to_datetime(daily["date_local"]).dt.strftime("%Y-%m-%d")
    return daily

def to_daily_from_energy(df: pd.DataFrame) -> pd.DataFrame | None:
    # Already daily aggregates: a date column + an energy column (kWh or MWh)
    dcol = pick(df.columns, TIME_CANDS)
    ecol = pick(df.columns, ENERGY_DAILY_CANDS)
    if not dcol or not ecol:
        # Try: first col is date, 4th is energy (common PVDAQ daily export)
        if len(df.columns) >= 4:
            dcol = dcol or df.columns[0]
            ecol = ecol or df.columns[3]
        else:
            return None
    out = pd.DataFrame({
        "date_local": pd.to_datetime(df[dcol], errors="coerce"),
        "energy": pd.to_numeric(df[ecol], errors="coerce")
    }).dropna()
    if out.empty: return None
    # If values look like MWh, convert to kWh
    med = out["energy"].median()
    if np.isfinite(med) and med < 200:  # many MW plants have ~10–50 MWh/day
        # assume MWh → convert to kWh
        energy_kWh = out["energy"] * 1000.0
    else:
        # assume already kWh
        energy_kWh = out["energy"]
    out = pd.DataFrame({
        "date_local": out["date_local"].dt.strftime("%Y-%m-%d"),
        "energy_kWh": energy_kWh
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder with PVDAQ CSVs (e.g., pvdaq_system_9068)")
    ap.add_argument("--site", required=True, help="Label to write as 'site' in output CSV (e.g., PVDAQ9068)")
    ap.add_argument("--tz", default="America/Denver", help="Local timezone for daily aggregation (default America/Denver)")
    args = ap.parse_args()

    indir = Path(args.indir)
    outs = []
    for p in sorted(indir.rglob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        d = to_daily_from_energy(df)
        if d is None:
            d = to_daily_from_power(df, tz=args.tz)
        if d is not None and not d.empty:
            outs.append(d)

    if not outs:
        raise SystemExit("No usable CSVs found. Consider adjusting column candidates in the script.")

    daily = pd.concat(outs, ignore_index=True)
    daily = (daily.groupby("date_local")["energy_kWh"].sum()
             .reset_index().sort_values("date_local"))
    daily.insert(0, "site", args.site)
    out = Path("scada_daily_template.csv")
    daily.to_csv(out, index=False)
    print(f"Wrote {out} with {len(daily)} rows.")
if __name__ == "__main__":
    main()
