#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import pandas as pd
import numpy as np

def add_clipping_columns_from_ghi(df, pac_kw, dcac, eta_dc, eta_inv, dt_minutes, offset_deg):
    """
    Deterministic, first-order AC model from GHI with optional offpoint cosine derate.
    P_preAC_kw = (GHI/1000) * (PAC*DCAC) * eta_dc * eta_inv * cos(off)
    Then clip at PAC.
    """
    if "ghi_wm2" not in df.columns:
        # Try common alternates
        for alt in ("GHI", "ghi", "poa_global", "poa", "poa_wm2"):
            if alt in df.columns:
                df = df.rename(columns={alt:"ghi_wm2"})
                break
    if "ghi_wm2" not in df.columns:
        raise KeyError("Need a GHI/POA column (e.g., 'ghi_wm2' or 'poa_global').")

    ghi = pd.to_numeric(df["ghi_wm2"], errors="coerce").fillna(0.0).to_numpy()
    pac_kw = float(pac_kw)
    dcac = float(dcac)
    eta_dc = float(eta_dc)
    eta_inv = float(eta_inv)
    dt_h = float(dt_minutes) / 60.0
    cos_derate = math.cos(math.radians(abs(float(offset_deg))))

    plant_dc_kw = pac_kw * dcac
    p_pre = np.maximum(ghi / 1000.0, 0.0) * plant_dc_kw * eta_dc * eta_inv * cos_derate

    p_clip = np.clip(p_pre - pac_kw, 0.0, None)
    p_ac   = p_pre - p_clip

    e_clip = p_clip * dt_h
    e_ac   = p_ac   * dt_h

    out = df.copy()
    out["P_ac_raw_kw"]   = p_pre
    out["P_ac_clip_kw"]  = p_ac
    out["P_clip_kw"]     = p_clip
    out["E_clip_kwh"]    = e_clip
    out["E_ac_clip_kwh"] = e_ac
    return out, float(e_clip.sum()), float(e_ac.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--policy", required=True, choices=["baseline","offpoint","bin","annual"])
    ap.add_argument("--min_deg", type=float, default=0.0)
    ap.add_argument("--max_deg", type=float, default=0.0)
    ap.add_argument("--step_deg", type=float, default=1.0)
    ap.add_argument("--offset_deg", type=float, default=0.0)

    ap.add_argument("--p_ac_rated_kw", type=float, required=True, help="Plant AC nameplate (PAC) kW")
    ap.add_argument("--dcac_ratio", type=float, default=1.30, help="DC/AC ratio")
    ap.add_argument("--inverter_eff", type=float, default=0.96)
    ap.add_argument("--dc_system_eff", type=float, default=0.90)
    ap.add_argument("--dt_minutes", type=int, default=60)

    ap.add_argument("--write_hourly_parquet", type=str, default=None)
    ap.add_argument("--write_daily_csv", type=str, default=None)

    args = ap.parse_args()

    df = pd.read_csv(args.weather)

    df_hourly, clip_kwh, ac_kwh = add_clipping_columns_from_ghi(
        df,
        pac_kw=args.p_ac_rated_kw,
        dcac=args.dcac_ratio,
        eta_dc=args.dc_system_eff,
        eta_inv=args.inverter_eff,
        dt_minutes=args.dt_minutes,
        offset_deg=args.offset_deg
    )

    if args.write_hourly_parquet:
        p = Path(args.write_hourly_parquet)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            df_hourly.to_parquet(p)
        except Exception:
            df_hourly.to_csv(str(p).replace(".parquet", ".csv"), index=False)

    if args.write_daily_csv:
        out = Path(args.write_daily_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = not out.exists()
        with out.open("a") as fh:
            if header:
                fh.write("date,site,policy,clip_kWh_day,ac_kWh_day\n")
            fh.write(f"{args.date},{args.site},{args.policy},{clip_kwh},{ac_kwh}\n")

    print(json.dumps({
        "date": args.date,
        "site": args.site,
        "policy": args.policy,
        "clip_kWh_day": clip_kwh,
        "ac_kWh_day": ac_kwh
    }))

if __name__ == "__main__":
    main()
