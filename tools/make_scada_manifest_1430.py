#!/usr/bin/env python3
import re, sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(".")
SRC_DIR = BASE / "data_raw" / "validation_system_1430"
OUT = SRC_DIR / "scada_pvdaq1430_2017_noNov.csv"#!/usr/bin/env python3
import re, sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(".")
SRC_DIR = BASE / "data_raw" / "validation_system_1430"
OUT = SRC_DIR / "scada_pvdaq1430_2017_noNov.csv"
EXCLUDE_MONTHS = {98}  # exclude November

# Column candidates in PVDAQ exports
TIME_COLS  = ["measured_on", "time_local", "Timestamp"]
POWER_COLS = ["ac_power__5074", "P_AC_kW", "power_kw", "Pac_kW"]  # AC power

def parse_date_from_name(name: str) -> str:
    m = re.search(r"date_(\d{4})_(\d{2})_(\d{2})", name)
    if not m: raise ValueError(f"Cannot parse date from {name}")
    y, mm, dd = m.groups(); return f"{y}-{mm}-{dd}"

def power_to_kW(series: pd.Series) -> pd.Series:
    """Return AC power in kW; infer if values are W or kW by magnitude."""
    s = pd.to_numeric(series, errors="coerce")
    q95 = s.quantile(0.95)
    if pd.isna(q95): return s
    # If the plant is tens of kW, PVDAQ often logs AC power in W.
    # Heuristic: if 95th percentile > 2000, it's likely W → convert to kW.
    return s/1000.0 if q95 > 2000 else s

def integrate_day_kWh(df: pd.DataFrame):
    pcol = next((c for c in POWER_COLS if c in df.columns), None)
    if pcol is None: raise ValueError("No AC power column found")
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        # Fall back to fixed 15-min spacing if no timestamp
        p_kw = power_to_kW(df[pcol])
        return float((p_kw.fillna(0) * 0.25).sum()), pd.NaT, pd.NaT, int(len(p_kw)), f"sum15:{pcol}"

    # Integrate AC power vs actual timestamps (trapezoid)
    t = pd.to_datetime(df[tcol], errors="coerce")
    p_kw = power_to_kW(df[pcol])
    ok = (~t.isna()) & (~p_kw.isna())
    if ok.sum() < 2:
        return 0.0, t.min(), t.max(), int(ok.sum()), f"trapz:{pcol}@na"

    tt_h = (t[ok].astype("int64")/1e9)/3600.0
    pp   = p_kw[ok].astype(float).values
    e_kWh = float(np.trapz(y=pp, x=tt_h))
    return max(0.0, e_kWh), t.min(), t.max(), int(ok.sum()), f"trapz:{pcol}"

def main():
    files = sorted((SRC_DIR).glob("system_1430__date_*.csv"))
    if not files:
        print(f"No daily files in {SRC_DIR}", file=sys.stderr); sys.exit(1)

    rows = []
    for f in files:
        date_local = parse_date_from_name(f.name)
        if int(date_local[5:7]) in EXCLUDE_MONTHS:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[SKIP] {f.name}: read error {e}", file=sys.stderr); continue

        try:
            e_kWh, t0, t1, n, method = integrate_day_kWh(df)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr); continue

        rows.append({
            "site":"PVDAQ1430",
            "date_local":date_local,
            "start_local":t0,
            "end_local":t1,
            "samples":n,
            "energy_kWh":round(e_kWh,3),
            "energy_source":method
        })

    if not rows:
        print("No rows extracted.", file=sys.stderr); sys.exit(2)

    out = pd.DataFrame(rows).sort_values("date_local").reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote: {OUT}  (rows={len(out)})")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()


# Column candidates in PVDAQ exports
TIME_COLS  = ["measured_on", "time_local", "Timestamp"]
POWER_COLS = ["ac_power__5074", "P_AC_kW", "power_kw", "Pac_kW"]  # AC power

def parse_date_from_name(name: str) -> str:
    m = re.search(r"date_(\d{4})_(\d{2})_(\d{2})", name)
    if not m: raise ValueError(f"Cannot parse date from {name}")
    y, mm, dd = m.groups(); return f"{y}-{mm}-{dd}"

def power_to_kW(series: pd.Series) -> pd.Series:
    """Return AC power in kW; infer if values are W or kW by magnitude."""
    s = pd.to_numeric(series, errors="coerce")
    q95 = s.quantile(0.95)
    if pd.isna(q95): return s
    # If the plant is tens of kW, PVDAQ often logs AC power in W.
    # Heuristic: if 95th percentile > 2000, it's likely W → convert to kW.
    return s/1000.0 if q95 > 2000 else s

def integrate_day_kWh(df: pd.DataFrame):
    pcol = next((c for c in POWER_COLS if c in df.columns), None)
    if pcol is None: raise ValueError("No AC power column found")
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        # Fall back to fixed 15-min spacing if no timestamp
        p_kw = power_to_kW(df[pcol])
        return float((p_kw.fillna(0) * 0.25).sum()), pd.NaT, pd.NaT, int(len(p_kw)), f"sum15:{pcol}"

    # Integrate AC power vs actual timestamps (trapezoid)
    t = pd.to_datetime(df[tcol], errors="coerce")
    p_kw = power_to_kW(df[pcol])
    ok = (~t.isna()) & (~p_kw.isna())
    if ok.sum() < 2:
        return 0.0, t.min(), t.max(), int(ok.sum()), f"trapz:{pcol}@na"

    tt_h = (t[ok].astype("int64")/1e9)/3600.0
    pp   = p_kw[ok].astype(float).values
    e_kWh = float(np.trapz(y=pp, x=tt_h))
    return max(0.0, e_kWh), t.min(), t.max(), int(ok.sum()), f"trapz:{pcol}"

def main():
    files = sorted((SRC_DIR).glob("system_1430__date_*.csv"))
    if not files:
        print(f"No daily files in {SRC_DIR}", file=sys.stderr); sys.exit(1)

    rows = []
    for f in files:
        date_local = parse_date_from_name(f.name)
        if int(date_local[5:7]) in EXCLUDE_MONTHS:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[SKIP] {f.name}: read error {e}", file=sys.stderr); continue

        try:
            e_kWh, t0, t1, n, method = integrate_day_kWh(df)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr); continue

        rows.append({
            "site":"PVDAQ1430",
            "date_local":date_local,
            "start_local":t0,
            "end_local":t1,
            "samples":n,
            "energy_kWh":round(e_kWh,3),
            "energy_source":method
        })

    if not rows:
        print("No rows extracted.", file=sys.stderr); sys.exit(2)

    out = pd.DataFrame(rows).sort_values("date_local").reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote: {OUT}  (rows={len(out)})")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
