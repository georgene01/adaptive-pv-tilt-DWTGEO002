import argparse, pandas as pd, pathlib
from zoneinfo import ZoneInfo

def _read_power_csv_autoheader(path):
    import pandas as pd, io, codecs
    with open(path, 'rb') as f:
        raw = f.read()
    # Handle potential UTF-8 BOM gracefully
    text = codecs.decode(raw, 'utf-8', errors='replace')
    lines = text.splitlines()
    # Find header row: prefer a line starting with YEAR, otherwise the first comma-rich line
    hdr_idx = None
    for i, line in enumerate(lines):
        L = line.strip()
        if L.startswith("YEAR,") or L.startswith("YEAR ,"):
            hdr_idx = i
            break
    if hdr_idx is None:
        # fallback: first line that contains at least 3 commas
        for i, line in enumerate(lines):
            if line.count(",") >= 3 and all(x.isalpha() or x in "_, " for x in line.replace(",","")):
                hdr_idx = i
                break
    if hdr_idx is None:
        # last resort: try letting pandas sniff; will likely error if metadata remains
        return pd.read_csv(io.StringIO(text), low_memory=False)
    # Slice text from header onward
    cleaned = "\n".join(lines[hdr_idx:])
    df = pd.read_csv(io.StringIO(cleaned), low_memory=False)
    return df


def _ensure_time_utc(df, tz_str):
    import pandas as pd
    # if already present, do nothing
    if 'time_utc' in df.columns:
        return df
    # normalize column names for detection
    cols = {c.lower(): c for c in df.columns}
    # POWER-style YEAR,MO,DA,HR
    if all(k in cols for k in ('year','mo','da','hr')):
        y = df[cols['year']].astype(int)
        m = df[cols['mo']].astype(int)
        d = df[cols['da']].astype(int)
        h = df[cols['hr']].astype(int)
        dt = pd.to_datetime({'year': y, 'month': m, 'day': d, 'hour': h}, errors='coerce')
        # local timestamps -> localize -> convert to UTC
        dt = dt.dt.tz_localize(tz_str, nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert('UTC')
        df['time_utc'] = dt.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        return df
    # Fallback: common timestamp columns
    for cand in ('time','timestamp','date_time','datetime'):
        if cand in cols:
            s = pd.to_datetime(df[cols[cand]], errors='coerce')
            try:
                s = s.dt.tz_localize(tz_str, nonexistent='shift_forward', ambiguous='NaT')
            except Exception:
                # already tz-aware or not a Series with .dt
                pass
            try:
                s = s.dt.tz_convert('UTC')
            except Exception:
                pass
            df['time_utc'] = pd.to_datetime(s, errors='coerce').dt.tz_convert('UTC').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            return df
    raise ValueError("Could not synthesize 'time_utc' from available columns.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)          # e.g. NC
    ap.add_argument("--in_hourly", required=True)     # e.g. data_raw/NC_2024_POWER.csv
    ap.add_argument("--tz", required=True)            # e.g. Africa/Johannesburg
    ap.add_argument("--outdir", default="tmp/daily_weather")
    args = ap.parse_args()
    df = _read_power_csv_autoheader(args.in_hourly)
    df = _ensure_time_utc(df, args.tz)
    if 'time_utc' not in df.columns:
        raise SystemExit("Expected a 'time_utc' column (UTC timestamps).")

    ts_utc = pd.to_datetime(df['time_utc'], utc=True, errors='coerce')
    if ts_utc.isna().any():
        bad = df.loc[ts_utc.isna(), 'time_utc'].head().tolist()
        raise SystemExit(f"Unparseable UTC times: {bad}")

    tz = ZoneInfo(args.tz)
    ts_local = ts_utc.dt.tz_convert(tz)
    df['date_local'] = ts_local.dt.date.astype(str)

    root = pathlib.Path(args.outdir) / args.site
    root.mkdir(parents=True, exist_ok=True)
    for day, sub in df.groupby('date_local'):
        (root / f"{args.site}_{day}.csv").write_text(sub.to_csv(index=False))

