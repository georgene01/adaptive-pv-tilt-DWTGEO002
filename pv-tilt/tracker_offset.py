#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.tracking import singleaxis as pv_singleaxis

try:
    import requests
except Exception:
    requests = None

# --------- Defaults ----------
TZ = "Africa/Johannesburg"
ALBEDO = 0.2
U0, U1 = 25.0, 6.84          # Faiman
GAMMA_P = -0.0045            # -0.45%/°C
P_STC_W = 440.0
MAX_TILT = 60.0

# --------- IO ----------
def fetch_nasa_power(lat: float, lon: float, date_str: str, tz: str) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("Install requests: pip install requests")
    y, m, d = date_str.split("-")
    start = end = f"{y}{m}{d}"
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": lat, "longitude": lon, "start": start, "end": end,
        "community": "RE", "time-standard": "UTC",
        "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS2M",
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    times = sorted(data["T2M"].keys())
    rows = []
    for t in times:
        rows.append({
            "time": pd.to_datetime(t, format="%Y%m%d%H", utc=True).tz_convert(tz),
            "GHI":  data["ALLSKY_SFC_SW_DWN"][t],
            "DNI":  data["ALLSKY_SFC_SW_DNI"][t],
            "DHI":  data["ALLSKY_SFC_SW_DIFF"][t],
            "Tamb": data["T2M"][t],
            "Wspd": data["WS2M"][t],
        })
    return pd.DataFrame(rows).set_index("time").astype(float)

# --------- Models ----------
def _fill_dni_dhi(df: pd.DataFrame, site: Location):
    ghi = df["GHI"].copy(); dni = df["DNI"].copy(); dhi = df["DHI"].copy()
    sp  = site.get_solarposition(df.index)
    if dni.isna().any() or dhi.isna().any():
        dni_disc = pvlib.irradiance.disc(ghi, sp["zenith"], df.index)["dni"]
        kt = pvlib.clearsky.ineichen_clearsky_index(ghi, sp["apparent_zenith"])
        dhi_erbs = pvlib.irradiance.erbs(ghi, kt, sp["zenith"])["dhi"]
        dni = dni.fillna(dni_disc); dhi = dhi.fillna(dhi_erbs)
    return dni, dhi, ghi, sp

def poa_perez_series(df: pd.DataFrame, site: Location, tilt_s: pd.Series, azm_s: pd.Series, albedo: float):
    dni, dhi, ghi, sp = _fill_dni_dhi(df, site)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_s.values, surface_azimuth=azm_s.values,
        dni=dni.values, ghi=ghi.values, dhi=dhi.values,
        solar_zenith=sp["apparent_zenith"].values, solar_azimuth=sp["azimuth"].values,
        model="perez", albedo=albedo, dni_extra=dni_extra.values
    )
    return pd.Series(poa["poa_global"], index=df.index).clip(lower=0.0)

def tcell_faiman(df: pd.DataFrame, gpoa: pd.Series, u0: float, u1: float):
    w = np.maximum(df["Wspd"], 0.1)
    return df["Tamb"] + gpoa / (u0 + u1*w)

def tcell_noct(df: pd.DataFrame, gpoa: pd.Series, noct_c: float):
    return df["Tamb"] + (noct_c - 20.0) * (gpoa / 800.0)

def pdc(gpoa: pd.Series, tcell: pd.Series, p_stc: float, gamma_p: float):
    eff = np.maximum(1.0 + gamma_p * (tcell - 25.0), 0.0)
    return p_stc * (gpoa / 1000.0) * eff

# --------- Orientations ----------
def orientation_series(index, site: Location, mode: str,
                       axis_azm=0.0, axis_tilt=0.0, max_angle=MAX_TILT,
                       backtrack=False, gcr=0.3,
                       fixed_tilt=30.0, fixed_azm=0.0) -> Tuple[pd.Series, pd.Series]:
    sp = site.get_solarposition(index)
    if mode == "fixed":
        return pd.Series(fixed_tilt, index=index), pd.Series(fixed_azm, index=index)
    if mode == "dualaxis":
        tilt = (90.0 - sp["apparent_zenith"]).clip(0, 90); azm = sp["azimuth"]
        return tilt, azm
    if mode == "singleaxis":
        trk = pv_singleaxis(sp["apparent_zenith"], sp["azimuth"],
                            axis_tilt=axis_tilt, axis_azimuth=axis_azm,
                            max_angle=max_angle, backtrack=backtrack, gcr=gcr)
        tilt = trk["surface_tilt"].fillna(0.0).clip(0, 90)
        azm  = trk["surface_azimuth"].fillna(0.0)
        return tilt, azm
    raise ValueError("mode must be fixed | dualaxis | singleaxis")

# --------- Offset optimizer ----------
def optimize_offset(df: pd.DataFrame, site: Location,
                    tilt_trk: pd.Series, azm_trk: pd.Series,
                    albedo: float,
                    p_stc: float, gamma_p: float,
                    u0: float, u1: float, noct_c: float, use_faiman: bool,
                    tilt_min: float, tilt_max: float,
                    off_start: int, off_stop: int, off_step: int) -> pd.DataFrame:
    offs = np.arange(off_start, off_stop + np.sign(off_step), off_step, dtype=float)
    rows = []
    for t in df.index:
        base = float(tilt_trk.loc[t]); az  = float(azm_trk.loc[t])
        best = (-1.0, 0.0, 0.0, float(df.loc[t,"Tamb"]))  # (P, Δ, G, Tc)
        for d in offs:
            tilt_t = float(np.clip(base + d, tilt_min, tilt_max))
            gpoa_t = poa_perez_series(df.loc[[t]], site,
                                      pd.Series([tilt_t], index=[t]),
                                      pd.Series([az], index=[t]), albedo)
            if use_faiman:
                w = max(float(df.loc[t,"Wspd"]), 0.1)
                tc = float(df.loc[t,"Tamb"]) + float(gpoa_t.iloc[0]) / (u0 + u1*w)
            else:
                tc = float(df.loc[t,"Tamb"]) + (noct_c - 20.0) * (float(gpoa_t.iloc[0]) / 800.0)
            p  = p_stc * (float(gpoa_t.iloc[0]) / 1000.0) * max(1.0 + gamma_p*(tc - 25.0), 0.0)
            if p > best[0]:
                best = (p, float(d), float(gpoa_t.iloc[0]), float(tc))
        rows.append({"time": t, "offset_deg": best[1], "Gpoa_opt": best[2], "Tcell_opt": best[3], "Pdc_W_opt": best[0]})
    return pd.DataFrame(rows).set_index("time")

# --------- Runner ----------
def run_day(date: str, lat: float, lon: float,
            mode: str,
            axis_azm: float, axis_tilt: float, max_angle: float, backtrack: bool, gcr: float,
            fixed_tilt: float, fixed_azm: float,
            use_faiman: bool, noct_c: float,
            offset_sweep: bool, offset_range: Tuple[int,int,int],
            tilt_bounds: Tuple[float,float],
            p_stc: float, gamma_p: float, albedo: float,
            modules: int) -> Dict[str,object]:

    site = Location(lat, lon, TZ, 0, "site")
    wx = fetch_nasa_power(lat, lon, date, TZ)

    # tracking / baseline
    tilt_trk, azm_trk = orientation_series(wx.index, site, mode,
                                           axis_azm, axis_tilt, max_angle, backtrack, gcr,
                                           fixed_tilt, fixed_azm)
    gpoa_trk = poa_perez_series(wx, site, tilt_trk, azm_trk, albedo)
    tcell_trk = tcell_faiman(wx, gpoa_trk, U0, U1) if use_faiman else tcell_noct(wx, gpoa_trk, noct_c)
    pdc_trk = pdc(gpoa_trk, tcell_trk, p_stc, gamma_p) * modules
    E_trk = float(pdc_trk.sum())

    results = {
        "timeseries_base": pd.DataFrame({
            "GHI": wx["GHI"], "DNI": wx["DNI"], "DHI": wx["DHI"],
            "Tamb_C": wx["Tamb"], "Wspd_mps": wx["Wspd"],
            "tilt_deg": tilt_trk, "azimuth_deg": azm_trk,
            "Gpoa_Wm2": gpoa_trk, "Tcell_C": tcell_trk, "Pdc_W_total": pdc_trk
        }),
        "E_Wh_tracking": E_trk
    }

    if offset_sweep:
        omin, omax, ostep = offset_range
        tmin, tmax = tilt_bounds
        opt = optimize_offset(wx, site, tilt_trk, azm_trk, albedo, p_stc, gamma_p,
                              U0, U1, noct_c, use_faiman, tmin, tmax, omin, omax, ostep)
        p_opt = opt["Pdc_W_opt"].astype(float) * modules
        results.update({
            "timeseries_opt": pd.DataFrame({
                "GHI": wx["GHI"], "DNI": wx["DNI"], "DHI": wx["DHI"],
                "Tamb_C": wx["Tamb"], "Wspd_mps": wx["Wspd"],
                "tilt_deg": (tilt_trk + opt["offset_deg"]).clip(tmin, tmax),
                "azimuth_deg": azm_trk,
                "offset_deg": opt["offset_deg"],
                "Gpoa_Wm2": opt["Gpoa_opt"], "Tcell_C": opt["Tcell_opt"],
                "Pdc_W_total": p_opt
            }),
            "offsets": opt[["offset_deg","Pdc_W_opt"]],
            "E_Wh_opt": float(p_opt.sum()),
        })
        results["delta_Wh"] = results["E_Wh_opt"] - results["E_Wh_tracking"]
        results["delta_pct"] = 100.0*(results["E_Wh_opt"]/results["E_Wh_tracking"] - 1.0)

    return results

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracking vs thermal-aware off-tilt (NASA POWER).")
    ap.add_argument("--date", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--mode", choices=["dualaxis","singleaxis","fixed"], default="dualaxis")
    ap.add_argument("--axis-azm", type=float, default=0.0)
    ap.add_argument("--axis-tilt", type=float, default=0.0)
    ap.add_argument("--max-angle", type=float, default=MAX_TILT)
    ap.add_argument("--backtrack", action="store_true")
    ap.add_argument("--gcr", type=float, default=0.30)
    ap.add_argument("--fixed-tilt", type=float, default=30.0)
    ap.add_argument("--fixed-azm", type=float, default=0.0)

    ap.add_argument("--offset-sweep", action="store_true")
    ap.add_argument("--offset-range", type=str, default="-20,20,1")  # Δmin,Δmax,step
    ap.add_argument("--tilt-bounds",  type=str, default="0,60")
    ap.add_argument("--noct", type=float, default=45.0)
    ap.add_argument("--faiman", action="store_true")
    ap.add_argument("--albedo", type=float, default=ALBEDO)
    ap.add_argument("--p-stc", type=float, default=P_STC_W)
    ap.add_argument("--gamma", type=float, default=GAMMA_P)
    ap.add_argument("--modules", type=int, default=1)

    args = ap.parse_args(argv)
    omin, omax, ostep = (int(x) for x in args.offset_range.split(","))
    tmin, tmax = (float(x) for x in args.tilt_bounds.split(","))

    res = run_day(
        date=args.date, lat=args.lat, lon=args.lon,
        mode=args.mode,
        axis_azm=args.axis_azm, axis_tilt=args.axis_tilt, max_angle=args.max_angle,
        backtrack=bool(args.backtrack), gcr=args.gcr,
        fixed_tilt=args.fixed_tilt, fixed_azm=args.fixed_azm,
        use_faiman=bool(args.faiman), noct_c=args.noct,
        offset_sweep=bool(args.offset_sweep), offset_range=(omin, omax, ostep),
        tilt_bounds=(tmin, tmax),
        p_stc=args.p_stc, gamma_p=args.gamma, albedo=args.albedo,
        modules=args.modules
    )

    tag = f"{args.date}_{args.mode}"
    res["timeseries_base"].to_csv(f"timeseries_{tag}_base.csv")
    if "timeseries_opt" in res:
        res["timeseries_opt"].to_csv(f"timeseries_{tag}_opt_offset.csv")
        res["offsets"].to_csv(f"offsets_{tag}.csv")
    summ = {
        "date": args.date, "lat": args.lat, "lon": args.lon, "mode": args.mode,
        "E_Wh_tracking": res["E_Wh_tracking"], "E_Wh_opt": res.get("E_Wh_opt"),
        "delta_Wh": res.get("delta_Wh"), "delta_pct": res.get("delta_pct")
    }
    with open(f"summary_{tag}.json","w") as f: json.dump(summ, f, indent=2)
    print(f"OK | {summ}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
