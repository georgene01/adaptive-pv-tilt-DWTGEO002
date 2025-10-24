# Weather QC script (compatible with hourly-safe POWER columns).
# Usage (one line):
#   python tools/qc_weather.py --in data_raw/NC_2024_POWER.csv --out qc/NC_2024_qc_report.json --timeshift qc/NC_2024_timeshift_check.txt

import argparse, json, pandas as pd
import numpy as np

IRR_COLS = ["ghi_wm2"]              # hourly-safe fetcher only has GHI
NUM_COLS = ["ghi_wm2","temp_air_c","wind_speed_ms","rel_humidity_pct","pressure_pa","albedo"]

def qc_dataframe(df: pd.DataFrame, interpolate_small_gaps=True):
    report = {"input_rows": int(len(df))}

    # Ensure expected cols exist
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Drop rows where all irradiances are NaN (here: just GHI)
    mask_all_nan = df[IRR_COLS].isna().all(axis=1)
    report["dropped_all_irr_nan"] = int(mask_all_nan.sum())
    df = df.loc[~mask_all_nan].copy()

    # Clamp negative irradiance to 0
    for c in IRR_COLS:
        neg = (df[c].fillna(0) < 0).sum()
        report[f"clamped_neg_{c}"] = int(neg)
        df.loc[df[c] < 0, c] = 0.0

    # Bounds
    def clamp_bounds(col, lo=None, hi=None):
        n = 0
        if lo is not None:
            n += int((df[col].dropna() < lo).sum())
            df.loc[df[col] < lo, col] = lo
        if hi is not None:
            n += int((df[col].dropna() > hi).sum())
            df.loc[df[col] > hi, col] = hi
        return n

    report["bounded_temp_outliers"] = clamp_bounds("temp_air_c", -40, 60)
    report["bounded_wind_outliers"] = clamp_bounds("wind_speed_ms", 0, None)
    report["bounded_rh_outliers"]   = clamp_bounds("rel_humidity_pct", 0, 100)

    # Optional interpolation for single missing hours
    if interpolate_small_gaps:
        before = df.isna().sum().to_dict()
        df = df.sort_values("time_utc").reset_index(drop=True)
        cols_to_interp = ["ghi_wm2","temp_air_c","wind_speed_ms","rel_humidity_pct","pressure_pa","albedo"]
        df[cols_to_interp] = df[cols_to_interp].interpolate(limit=1)
        after = df.isna().sum().to_dict()
        report["interpolated_counts"] = {k:int(before[k])-int(after[k]) for k in before.keys()}
    else:
        report["interpolated_counts"] = {}

    report["output_rows"] = int(len(df))
    return df, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeshift", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["time_utc"])
    df, rep = qc_dataframe(df, interpolate_small_gaps=True)

    # Save QC report
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    # Clean CSV alongside (same name with _qc.csv)
    out_csv = args.inp.replace(".csv","_qc.csv")
    df.to_csv(out_csv, index=False)

    # Timeshift checklist
    with open(args.timeshift, "w") as f:
        f.write(
            "Timeshift sanity checklist (UTC -> Africa/Johannesburg):\n"
            "- Confirm solar noon (UTC) +2h aligns with expected SAST solar noon.\n"
            "- Plot GHI diurnal curve; check sunrise/sunset transitions.\n"
            "- If offset seen, shift timestamps accordingly before analysis.\n"
        )

    print(f"QC report: {args.out}")
    print(f"Clean CSV : {out_csv}")
    print(f"Wrote timeshift checklist: {args.timeshift}")

if __name__ == "__main__":
    main()
