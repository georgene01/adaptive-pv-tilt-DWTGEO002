import argparse, pathlib, re
import pandas as pd
import numpy as np

def parse_iso_like(series):
    try:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
        if ts.notna().mean() > 0.8:
            # If parsed datetimes are timezone-naive (rare), localize to UTC
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            return ts
    except Exception:
        pass
    return None

def detect_timestamp(df):
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) Single datetime-ish column
    candidates = [
        k for k in cols_lower
        if any(tag in k for tag in ["time", "timestamp", "datetime", "date_time", "utc"])
        or re.fullmatch(r".*date.*", k)  # sometimes a full ISO sits in a "date" column
    ]
    # Put likely ones first
    prefer = ["timestamp","time_utc","utc_time","datetime","date_time","time","date"]
    candidates = sorted(set(candidates), key=lambda x: (prefer.index(x) if x in prefer else 999, x))
    for k in candidates:
        ts = parse_iso_like(df[cols_lower[k]])
        if ts is not None:
            return ts

    # Helper to find any of several names
    def pick(*names):
        for n in names:
            if n in cols_lower: return cols_lower[n]
        return None

    # 2) date + hour
    date_col = pick("date","day","yyyymmdd","yyyy-mm-dd")
    hour_col = pick("hr","hour","hh","h")
    if date_col is not None and hour_col is not None:
        date_vals = df[date_col]
        # support yyyymmdd integer or YYYY-MM-DD string
        date_vals = pd.to_datetime(date_vals.astype(str), errors="coerce", utc=True)
        hour_vals = pd.to_numeric(df[hour_col], errors="coerce")
        ts = date_vals + pd.to_timedelta(hour_vals.fillna(0), unit="h")
        if ts.notna().mean() > 0.8:
            # ensure tz-aware UTC
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")
            return ts

    # 3) yyyymmdd (as int) + hour even if not labeled "date"
    ymd_like = None
    for c in df.columns:
        s = df[c]
        # Try columns that look like 8-digit yyyymmdd
        if np.issubdtype(s.dtype, np.number) or s.dtype == object:
            s_str = s.astype(str).str.strip()
            mask = s_str.str.fullmatch(r"\d{8}")
            if mask.mean() > 0.8:
                ymd_like = c
                break
    if ymd_like is not None and hour_col is not None:
        date_vals = pd.to_datetime(df[ymd_like].astype(str), format="%Y%m%d", errors="coerce", utc=True)
        hour_vals = pd.to_numeric(df[hour_col], errors="coerce")
        ts = date_vals + pd.to_timedelta(hour_vals.fillna(0), unit="h")
        if ts.notna().mean() > 0.8:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")
            return ts

    # 4) year + doy + hour
    year_col = pick("year")
    doy_col  = pick("doy","day_of_year","julian")
    if year_col is not None and doy_col is not None:
        try:
            year_vals = pd.to_numeric(df[year_col], errors="coerce")
            doy_vals  = pd.to_numeric(df[doy_col], errors="coerce")
            hour_vals = pd.to_numeric(df[hour_col] if hour_col else 0, errors="coerce")
            base = pd.to_datetime(year_vals.astype("Int64").astype(str), format="%Y", errors="coerce", utc=True)
            ts = base + pd.to_timedelta(doy_vals.fillna(1)-1, unit="D") + pd.to_timedelta(hour_vals.fillna(0), unit="h")
            if ts.notna().mean() > 0.8:
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize("UTC")
                else:
                    ts = ts.dt.tz_convert("UTC")
                return ts
        except Exception:
            pass

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--in", dest="infile", required=True)   # e.g., data_raw/NC_2024_POWER.csv
    ap.add_argument("--tz", required=True)                  # e.g., Africa/Johannesburg
    ap.add_argument("--out", required=True)                 # e.g., data_raw/NC_2024_POWER_qc_LOCAL.csv
    args = ap.parse_args()

    p = pathlib.Path(args.infile)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")

    df = pd.read_csv(p, low_memory=False)
    ts_utc = detect_timestamp(df)
    if ts_utc is None or ts_utc.notna().sum() == 0:
        print("Columns in file:\n  " + "\n  ".join(map(str, df.columns)))
        raise SystemExit("Could not detect a UTC timestamp from the file. "
                         "If you share the first 5 lines/column names, I can add that pattern.")

    # convert to desired local tz
    try:
        ts_local = ts_utc.dt.tz_convert(args.tz)
    except Exception:
        # if args.tz is an alias, this will still work with zoneinfo names
        ts_local = ts_utc.dt.tz_convert("Africa/Johannesburg")

    # Local helpers
    df["ts_utc"]      = ts_utc
    df["ts_local"]    = ts_local
    df["date_local"]  = ts_local.dt.date.astype(str)
    df["year_local"]  = ts_local.dt.year
    df["month_local"] = ts_local.dt.month
    df["day_local"]   = ts_local.dt.day
    df["hour_local"]  = ts_local.dt.hour

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"Wrote LOCAL file: {outp}  rows={len(df)}")

if __name__ == "__main__":
    main()
