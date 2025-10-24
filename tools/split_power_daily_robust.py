#!/usr/bin/env python3
import argparse, pathlib
import pandas as pd

def read_power_any(path):
    p = pathlib.Path(path)
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        header_idx = None
        for i, L in enumerate(lines[:120]):
            # find a plausible CSV header
            if ("," in L) and any(k in L for k in ("YEAR","MO","DA","DY","HR")):
                header_idx = i
                break
        if header_idx is None:
            raise SystemExit("Could not locate a header row containing YEAR/MO/DA(or DY)/HR.")
        from io import StringIO
        buf = StringIO("".join(lines[header_idx:]))
        return pd.read_csv(buf, low_memory=False)

def _pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    ap = argparse.ArgumentParser(description="POWER → daily files (handles hourly or daily inputs)")
    ap.add_argument("--site", required=True, help="Output filename prefix (e.g., NC or WC)")
    ap.add_argument("--in_hourly", required=True, help="Input POWER CSV (hourly OR daily)")
    ap.add_argument("--outdir", required=True, help="Output directory root")
    args = ap.parse_args()

    df = read_power_any(args.in_hourly)

    col_Y = _pick(df, "YEAR", "year")
    col_M = _pick(df, "MO", "month")
    col_DA = _pick(df, "DA", "DY", "day")  # accept DA or DY
    col_HR = _pick(df, "HR", "hour")

    if not all([col_Y, col_M, col_DA]):
        raise SystemExit(f"Missing required date columns; have: {list(df.columns)[:20]}")

    # Build date string
    y = df[col_Y].astype(int)
    m = df[col_M].astype(int)
    d = df[col_DA].astype(int)
    df["date_local"] = (y.astype(str) + "-" +
                        m.astype(str).str.zfill(2) + "-" +
                        d.astype(str).str.zfill(2))

    # Output directory like tmp/daily_weather/NC_2023 or WC_2023 (caller decides)
    root = pathlib.Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    # Group by day and write
    n_files = 0
    for day, sub in df.groupby("date_local", sort=True):
        out = root / f"{args.site}_{day}.csv"
        out.write_text(sub.to_csv(index=False))
        n_files += 1
    print(f"[OK] Wrote {n_files} daily files under {root} with prefix '{args.site}_'")

if __name__ == "__main__":
    main()
