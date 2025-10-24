#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully integrated PV tilt simulator (improved)

Adds to your original:
  - NASA irradiance unit guard (kWh/m²·h -> W/m² if needed)
  - Accurate energy integration with real Δt (trapz)
  - Optional DC→AC scaling (inverter eff + DC wiring loss)
  - Validation vs measured CSV (MBE, RMSE, NRMSE, daily Wh)
  - Publication-ready plots (PDFs)

Outputs (same base files, plus plots and rich summary):
  - timeseries_<date>_<mode>_base.csv
  - timeseries_<date>_<mode>_opt_offset.csv
  - offsets_<date>_<mode>.csv
  - summary_<date>_<mode>.json
  - daily_power_comparison_<date>_<mode>.pdf
  - tilt_timeseries_<date>_<mode>.pdf
  - offset_timeseries_<date>_<mode>.pdf
  - energy_bar_day_<date>_<mode>.pdf
"""

from __future__ import annotations
import argparse, json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.tracking import singleaxis as pv_singleaxis

import matplotlib.pyplot as plt

try:
    import requests
except Exception:
    requests = None

# ---------------- Defaults ----------------
LAT_DEFAULT, LON_DEFAULT = -33.9249, 18.4241  # Cape Town
TZ_DEFAULT = "Africa/Johannesburg"
AZIMUTH_DEFAULT = 0.0
ALBEDO_DEFAULT = 0.2
P_STC_W_DEFAULT = 440.0
GAMMA_P_DEFAULT = -0.0045
NOCT_C_DEFAULT = 45.0               # °C
U0_DEFAULT, U1_DEFAULT = 25.0, 6.84 # Faiman
MAX_TILT_DEFAULT = 60.0

# ---------------- Utilities ----------------
def _ensure_wm2(df: pd.DataFrame) -> pd.DataFrame:
    """
    NASA POWER hourly irradiance sometimes arrives as kWh/m² per hour.
    If median GHI < 50, treat as kWh/m²·h and convert to W/m².
    """
    if "GHI" in df and df["GHI"].median() < 50:
        for k in ["GHI","DNI","DHI"]:
            if k in df:
                df[k] = df[k] * 1000.0
    return df

def integrate_energy_Wh(p_series: pd.Series) -> float:
    """Integrate power [W] over time to energy [Wh] with real Δt (trapz)."""
    if len(p_series) == 0:
        return 0.0
    # convert datetime index to seconds since epoch
    t = p_series.index.view(np.int64) / 1e9
    return float(np.trapezoid(p_series.values, t) / 3600.0)

def dc_to_ac(pdc: pd.Series, eta_inverter: float, dc_wiring_loss: float) -> pd.Series:
    """Apply simple DC→AC chain: AC = Pdc * (1 - loss) * eta."""
    eta_chain = max(0.0, min(1.0, (1.0 - dc_wiring_loss) * eta_inverter))
    return pdc * eta_chain

def validate_against_measured(sim_p: pd.Series, meas: pd.Series) -> dict:
    df = pd.DataFrame({"sim": sim_p, "meas": meas}).dropna()
    if df.empty:
        return {"note": "no overlapping timestamps"}
    mbe = float((df["sim"] - df["meas"]).mean())                      # W
    rmse = float(np.sqrt(((df["sim"] - df["meas"])**2).mean()))       # W
    nrmse = rmse / max(df["meas"].mean(), 1e-9)                       # unitless
    e_sim = integrate_energy_Wh(df["sim"])                            # Wh
    e_mea = integrate_energy_Wh(df["meas"])                           # Wh
    mbe_e = e_sim - e_mea                                             # Wh
    return {"MBE_W": mbe, "RMSE_W": rmse, "NRMSE": nrmse,
            "E_sim_Wh": e_sim, "E_meas_Wh": e_mea, "MBE_Wh": mbe_e}

def plot_day(results: Dict[str,object], tag: str):
    base = results["timeseries_base"]
    opt  = results.get("timeseries_opt")

    # 1) Power vs time
    plt.figure()
    base["Pdc_W_total"].plot(label="Perpendicular baseline")
    if opt is not None:
        opt["Pdc_W_total"].plot(label="Adaptive tilt (optimal)")
    plt.ylabel("Power [W]"); plt.xlabel("Time"); plt.legend(); plt.tight_layout()
    plt.savefig(f"daily_power_comparison_{tag}.pdf"); plt.close()

    # 2) Tilt vs time
    plt.figure()
    base["tilt_deg"].plot(label="Baseline tilt")
    if opt is not None:
        opt["tilt_deg"].plot(label="Optimal tilt")
    plt.ylabel("Tilt [deg]"); plt.xlabel("Time"); plt.legend(); plt.tight_layout()
    plt.savefig(f"tilt_timeseries_{tag}.pdf"); plt.close()

    # 3) Offset vs time (if any)
    if opt is not None and "offset_deg" in opt:
        plt.figure()
        opt["offset_deg"].plot(label="Δ*(t)")
        plt.ylabel("Offset [deg]"); plt.xlabel("Time"); plt.legend(); plt.tight_layout()
        plt.savefig(f"offset_timeseries_{tag}.pdf"); plt.close()

    # 4) Energy bar chart
    e_base = results["E_Wh_tracking"]
    e_opt  = results.get("E_Wh_opt")
    plt.figure()
    if e_opt is not None:
        plt.bar(["Baseline","Adaptive"], [e_base, e_opt])
    else:
        plt.bar(["Baseline"], [e_base])
    plt.ylabel("Daily Energy [Wh]"); plt.tight_layout()
    plt.savefig(f"energy_bar_day_{tag}.pdf"); plt.close()

# ---------------- Fetch ----------------
def fetch_nasa_power_direct(lat: float, lon: float, date_str: str, tz: str) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests not available. Install with: pip install requests")
    y, m, d = date_str.split("-")
    start = end = f"{y}{m}{d}"
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": lat, "longitude": lon,
        "start": start, "end": end, "community": "RE",
        "time-standard": "UTC",
        "parameters": ",".join([
            "ALLSKY_SFC_SW_DWN","ALLSKY_SFC_SW_DNI","ALLSKY_SFC_SW_DIFF",
            "T2M","WS2M"
        ]),
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
    df = pd.DataFrame(rows).set_index("time").astype(float)
    return _ensure_wm2(df)

def fetch_nasa_power_pvlib(lat: float, lon: float, date_str: str, tz: str) -> pd.DataFrame:
    df, _ = pvlib.iotools.get_nasa_power(
        lat, lon, start=date_str, end=date_str,
        parameters=['dni','dhi','ghi','temp_air','wind_speed'],
        community='re'
    )
    if not df.index.tz:
        df = df.tz_localize("UTC")
    df = df.tz_convert(tz).rename(columns={
        'ghi':'GHI','dni':'DNI','dhi':'DHI','temp_air':'Tamb','wind_speed':'Wspd'
    })
    df = df[['GHI','DNI','DHI','Tamb','Wspd']].astype(float)
    return _ensure_wm2(df)

# ---------------- Models ----------------
def fill_dni_dhi_if_needed(df: pd.DataFrame, site: Location) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ghi = df["GHI"].copy()
    dni = df["DNI"].copy()
    dhi = df["DHI"].copy()
    solpos = site.get_solarposition(df.index)

    if dni.isna().any() or dhi.isna().any():
        dni_disc = pvlib.irradiance.disc(ghi, solpos["zenith"], df.index)["dni"]
        kt = pvlib.clearsky.ineichen_clearsky_index(ghi, solpos["apparent_zenith"])
        dhi_erbs = pvlib.irradiance.erbs(ghi, kt, solpos["zenith"])["dhi"]
        dni = dni.fillna(dni_disc)
        dhi = dhi.fillna(dhi_erbs)

    return dni, dhi, ghi

def perez_poa_series(df: pd.DataFrame, site: Location,
                     tilt: pd.Series, azm: pd.Series, albedo: float) -> pd.Series:
    dni, dhi, ghi = fill_dni_dhi_if_needed(df, site)
    solpos = site.get_solarposition(df.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt.values,
        surface_azimuth=azm.values,
        dni=dni.values, ghi=ghi.values, dhi=dhi.values,
        solar_zenith=solpos["apparent_zenith"].values,
        solar_azimuth=solpos["azimuth"].values,
        model="perez",
        albedo=albedo,
        dni_extra=dni_extra.values
    )
    return pd.Series(poa["poa_global"], index=df.index).clip(lower=0.0)

def faiman_cell_temp(df: pd.DataFrame, gpoa: pd.Series, u0: float, u1: float) -> pd.Series:
    w = np.maximum(df["Wspd"], 0.1)
    return df["Tamb"] + gpoa / (u0 + u1 * w)

def noct_cell_temp(df: pd.DataFrame, gpoa: pd.Series, noct_c: float) -> pd.Series:
    return df["Tamb"] + (noct_c - 20.0) * (gpoa / 800.0)

def dc_power(gpoa: pd.Series, tcell: pd.Series, p_stc_w: float, gamma_p: float) -> pd.Series:
    eff = np.maximum(1.0 + gamma_p * (tcell - 25.0), 0.0)
    return p_stc_w * (gpoa / 1000.0) * eff

# ---------------- Orientations ----------------
def orientation_series(index: pd.DatetimeIndex, site: Location,
                       mode: str = "dualaxis",
                       axis_azimuth: float = 0.0,
                       axis_tilt: float = 0.0,
                       max_angle: float = 60.0,
                       backtrack: bool = False,
                       gcr: float = 0.3,
                       fixed_tilt: float = 30.0,
                       fixed_azm: float = AZIMUTH_DEFAULT) -> Tuple[pd.Series, pd.Series]:
    solpos = site.get_solarposition(index)
    if mode == "fixed":
        tilt = pd.Series(float(fixed_tilt), index=index)
        azm  = pd.Series(float(fixed_azm),  index=index)
    elif mode == "dualaxis":
        tilt = (90.0 - solpos["apparent_zenith"]).clip(lower=0.0, upper=90.0)
        azm  = solpos["azimuth"]
    elif mode == "singleaxis":
        trk = pv_singleaxis(
            apparent_zenith=solpos["apparent_zenith"],
            apparent_azimuth=solpos["azimuth"],
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            max_angle=max_angle,
            backtrack=backtrack,
            gcr=gcr
        )
        tilt = trk["surface_tilt"].fillna(0.0).clip(lower=0.0, upper=90.0)
        azm  = trk["surface_azimuth"].fillna(0.0)
    else:
        raise ValueError("mode must be one of: dualaxis | singleaxis | fixed")
    return tilt, azm

# ---------------- Off-tilt optimization ----------------
def optimize_offset_per_hour(df: pd.DataFrame, site: Location,
                             tilt_trk: pd.Series, azm_trk: pd.Series,
                             albedo: float,
                             p_stc: float, gamma_p: float,
                             u0: float, u1: float, noct_c: float,
                             use_faiman: bool,
                             tilt_bounds: Tuple[float,float],
                             offset_range: Tuple[int,int,int]) -> pd.DataFrame:
    off_start, off_stop, off_step = offset_range
    offsets = np.arange(off_start, off_stop + np.sign(off_step), off_step, dtype=float)
    out_rows = []
    for t in df.index:
        best_p, best_off, best_g, best_tc = -1.0, 0.0, 0.0, float(df.loc[t,"Tamb"])
        base_tilt = float(tilt_trk.loc[t]); azm_t = float(azm_trk.loc[t])
        for d in offsets:
            tilt_t = np.clip(base_tilt + d, tilt_bounds[0], tilt_bounds[1])
            gpoa_t = perez_poa_series(df.loc[[t]], site,
                                      pd.Series([tilt_t], index=[t]),
                                      pd.Series([azm_t],  index=[t]),
                                      albedo)
            if use_faiman:
                w = max(float(df.loc[t,"Wspd"]), 0.1)
                tcell_t = float(df.loc[t,"Tamb"]) + float(gpoa_t.iloc[0]) / (u0 + u1*w)
            else:
                tcell_t = float(df.loc[t,"Tamb"]) + (noct_c - 20.0) * (float(gpoa_t.iloc[0]) / 800.0)
            pdc_t = p_stc * (float(gpoa_t.iloc[0]) / 1000.0) * max(1.0 + gamma_p * (tcell_t - 25.0), 0.0)
            if pdc_t > best_p:
                best_p, best_off, best_g, best_tc = pdc_t, float(d), float(gpoa_t.iloc[0]), float(tcell_t)
        out_rows.append({"time": t, "offset_deg": best_off, "Gpoa_opt": best_g, "Tcell_opt": best_tc, "Pdc_W_opt": best_p})
    return pd.DataFrame(out_rows).set_index("time")

# ---------------- Runner ----------------
def run_day(
    date: str,
    lat: float = LAT_DEFAULT, lon: float = LON_DEFAULT, tz: str = TZ_DEFAULT,
    albedo: float = ALBEDO_DEFAULT,
    p_stc_w: float = P_STC_W_DEFAULT, gamma_p: float = GAMMA_P_DEFAULT,
    noct_c: float = NOCT_C_DEFAULT, u0: float = U0_DEFAULT, u1: float = U1_DEFAULT,
    use_faiman: bool = True,
    fetch_mode: str = "direct",
    orientation_mode: str = "dualaxis",
    fixed_tilt: float = 30.0, fixed_azm: float = AZIMUTH_DEFAULT,
    axis_azimuth: float = 0.0, axis_tilt: float = 0.0, max_angle: float = MAX_TILT_DEFAULT,
    backtrack: bool = False, gcr: float = 0.3,
    do_offset_sweep: bool = True, offset_range: Tuple[int,int,int] = (-20,20,1),
    tilt_bounds: Tuple[float,float] = (0.0, MAX_TILT_DEFAULT),
    n_modules: int = 1,
    eta_inverter: float = 0.96, dc_wiring_loss: float = 0.02,
    plotting: bool = True
) -> Dict[str, object]:

    site = Location(lat, lon, tz, 0, "site")

    # Fetch weather
    if fetch_mode == "direct":
        wx = fetch_nasa_power_direct(lat, lon, date, tz)
    elif fetch_mode == "pvlib":
        wx = fetch_nasa_power_pvlib(lat, lon, date, tz)
    else:
        raise ValueError("fetch_mode must be 'direct' or 'pvlib'")

    # Baseline orientation
    tilt_trk, azm_trk = orientation_series(
        wx.index, site,
        mode=orientation_mode,
        axis_azimuth=axis_azimuth, axis_tilt=axis_tilt,
        max_angle=max_angle, backtrack=backtrack, gcr=gcr,
        fixed_tilt=fixed_tilt, fixed_azm=fixed_azm
    )
    # Baseline POA/Temp/Power (DC)
    gpoa_trk = perez_poa_series(wx, site, tilt_trk, azm_trk, albedo)
    tcell_trk = faiman_cell_temp(wx, gpoa_trk, u0, u1) if use_faiman else noct_cell_temp(wx, gpoa_trk, noct_c)
    pdc_trk = dc_power(gpoa_trk, tcell_trk, p_stc_w, gamma_p) * n_modules

    # Offset optimization (Δ from tracking tilt)
    offsets_df = None
    if do_offset_sweep:
        offsets_df = optimize_offset_per_hour(
            df=wx, site=site, tilt_trk=tilt_trk, azm_trk=azm_trk,
            albedo=albedo, p_stc=p_stc_w, gamma_p=gamma_p,
            u0=u0, u1=u1, noct_c=noct_c, use_faiman=use_faiman,
            tilt_bounds=tilt_bounds, offset_range=offset_range
        )
        pdc_opt_series = offsets_df["Pdc_W_opt"].astype(float) * n_modules
    else:
        pdc_opt_series = None

    # Optional AC conversion for comparison/validation
    pac_trk = dc_to_ac(pdc_trk, eta_inverter, dc_wiring_loss)
    if pdc_opt_series is not None:
        pac_opt = dc_to_ac(pdc_opt_series, eta_inverter, dc_wiring_loss)
    else:
        pac_opt = None

    # Energy (Wh) with proper integration
    E_Wh_tracking = integrate_energy_Wh(pac_trk)
    E_Wh_opt = integrate_energy_Wh(pac_opt) if pac_opt is not None else None

    # Assemble outputs
    ts_base = pd.DataFrame({
        "GHI": wx["GHI"], "DNI": wx["DNI"], "DHI": wx["DHI"],
        "Tamb_C": wx["Tamb"], "Wspd_mps": wx["Wspd"],
        "tilt_deg": tilt_trk, "azimuth_deg": azm_trk,
        "Gpoa_Wm2": gpoa_trk, "Tcell_C": tcell_trk,
        "Pdc_W_total": pdc_trk, "Pac_W_total": pac_trk
    })
    results: Dict[str, object] = {
        "timeseries_base": ts_base,
        "E_Wh_tracking": E_Wh_tracking,
    }

    if pdc_opt_series is not None:
        ts_opt = pd.DataFrame({
            "GHI": wx["GHI"], "DNI": wx["DNI"], "DHI": wx["DHI"],
            "Tamb_C": wx["Tamb"], "Wspd_mps": wx["Wspd"],
            "tilt_deg": (tilt_trk + offsets_df["offset_deg"]).clip(lower=tilt_bounds[0], upper=tilt_bounds[1]),
            "azimuth_deg": azm_trk,
            "offset_deg": offsets_df["offset_deg"],
            "Gpoa_Wm2": offsets_df["Gpoa_opt"],
            "Tcell_C": offsets_df["Tcell_opt"],
            "Pdc_W_total": pdc_opt_series,
            "Pac_W_total": pac_opt
        })
        results.update({
            "timeseries_opt": ts_opt,
            "offsets": offsets_df[["offset_deg","Pdc_W_opt"]],
            "E_Wh_opt": E_Wh_opt,
            "delta_Wh": (E_Wh_opt - E_Wh_tracking) if E_Wh_opt is not None else None,
            "delta_pct": (100.0 * (E_Wh_opt / E_Wh_tracking - 1.0)) if E_Wh_opt is not None and E_Wh_tracking > 0 else None
        })

    # Plots
    if plotting:
        tag = f"{date}_{orientation_mode}"
        plot_day(results, tag)

    return results

# ---------------- CLI ----------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Improved PV tracking vs thermal-aware off-tilt simulator (NASA POWER).")
    ap.add_argument("--date", required=True, help="Local date YYYY-MM-DD")
    ap.add_argument("--lat", type=float, default=LAT_DEFAULT)
    ap.add_argument("--lon", type=float, default=LON_DEFAULT)
    ap.add_argument("--albedo", type=float, default=ALBEDO_DEFAULT)
    ap.add_argument("--fetch", choices=["direct","pvlib"], default="direct")

    # Electrical/thermal params
    ap.add_argument("--p-stc", type=float, default=P_STC_W_DEFAULT)
    ap.add_argument("--gamma", type=float, default=GAMMA_P_DEFAULT)
    ap.add_argument("--noct", type=float, default=NOCT_C_DEFAULT)
    ap.add_argument("--u0", type=float, default=U0_DEFAULT)
    ap.add_argument("--u1", type=float, default=U1_DEFAULT)
    ap.add_argument("--faiman", action="store_true", help="Use Faiman (default). If omitted, NOCT is used.")
    ap.add_argument("--modules", type=int, default=1)

    # Orientation
    ap.add_argument("--mode", choices=["dualaxis","singleaxis","fixed"], default="dualaxis",
                    help="Baseline orientation mode.")
    ap.add_argument("--fixed-tilt", type=float, default=30.0)
    ap.add_argument("--fixed-azm",  type=float, default=AZIMUTH_DEFAULT)

    # Single-axis tracker params
    ap.add_argument("--axis-azm",   type=float, default=0.0, help="Tracker axis azimuth (deg from North).")
    ap.add_argument("--axis-tilt",  type=float, default=0.0, help="Tracker axis tilt (deg).")
    ap.add_argument("--max-angle",  type=float, default=MAX_TILT_DEFAULT, help="Tracker maximum rotation (deg).")
    ap.add_argument("--backtrack",  action="store_true")
    ap.add_argument("--gcr",        type=float, default=0.3)

    # Offset sweep
    ap.add_argument("--offset-sweep", action="store_true", help="Enable thermal-aware off-tilt optimization.")
    ap.add_argument("--offset-range", type=str, default="-20,20,1", help="Δmin,Δmax,step in degrees, e.g. -20,20,1")
    ap.add_argument("--tilt-bounds",  type=str, default="0,60", help="Mechanical tilt bounds min,max (deg)")

    # AC + plotting + validation
    ap.add_argument("--eta-inverter", type=float, default=0.96)
    ap.add_argument("--dc-wiring-loss", type=float, default=0.02)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--meas-csv", type=str)
    ap.add_argument("--meas-col", type=str, default="P_AC_W")
    ap.add_argument("--meas-time-col", type=str, default="time")
    ap.add_argument("--meas-tz", type=str, default=TZ_DEFAULT)
    ap.add_argument("--meas-scale", type=float, default=1.0)

    args = ap.parse_args(argv)

    # Parse ranges
    try:
        omin, omax, ostep = (int(x) for x in args.offset_range.split(","))
    except Exception:
        raise SystemExit("Invalid --offset-range; expected 'start,stop,step' e.g. -20,20,1")
    try:
        tmin, tmax = (float(x) for x in args.tilt_bounds.split(","))
    except Exception:
        raise SystemExit("Invalid --tilt-bounds; expected 'min,max' e.g. 0,60")

    # Run
    res = run_day(
        date=args.date, lat=args.lat, lon=args.lon, tz=TZ_DEFAULT,
        albedo=args.albedo,
        p_stc_w=args.p_stc, gamma_p=args.gamma,
        noct_c=args.noct, u0=args.u0, u1=args.u1,
        use_faiman=bool(args.faiman), fetch_mode=args.fetch,
        orientation_mode=args.mode, fixed_tilt=args.fixed_tilt, fixed_azm=args.fixed_azm,
        axis_azimuth=args.axis_azm, axis_tilt=args.axis_tilt, max_angle=args.max_angle,
        backtrack=bool(args.backtrack), gcr=args.gcr,
        do_offset_sweep=bool(args.offset_sweep),
        offset_range=(omin, omax, ostep),
        tilt_bounds=(tmin, tmax),
        n_modules=args.modules,
        eta_inverter=args.eta_inverter, dc_wiring_loss=args.dc_wiring_loss,
        plotting=bool(args.plot)
    )

    tag = f"{args.date}_{args.mode}"

    # Write CSVs
    base = res["timeseries_base"]
    base.to_csv(f"timeseries_{tag}_base.csv")
    print(f"Wrote timeseries_{tag}_base.csv")

    if "timeseries_opt" in res:
        opt = res["timeseries_opt"]
        offs = res["offsets"]
        opt.to_csv(f"timeseries_{tag}_opt_offset.csv")
        offs.to_csv(f"offsets_{tag}.csv")
        print(f"Wrote timeseries_{tag}_opt_offset.csv")
        print(f"Wrote offsets_{tag}.csv")

    # Validation (optional)
    validation = None
    if args.meas_csv:
        try:
            mdf = pd.read_csv(args.meas_csv)
            tcol = args.meas_time_col
            mdf[tcol] = pd.to_datetime(mdf[tcol], errors="coerce")
            mdf = mdf.dropna(subset=[tcol]).set_index(tcol)

            # If timestamps are naive → localize; if already tz-aware → convert.
            if mdf.index.tz is None:
                mdf.index = mdf.index.tz_localize(args.meas_tz, nonexistent="shift_forward", ambiguous="NaT")
            else:
                mdf.index = mdf.index.tz_convert(args.meas_tz)

            # Finally convert to the model’s timezone
            mdf.index = mdf.index.tz_convert(TZ_DEFAULT)

            meas = (mdf[args.meas_col].astype(float) * args.meas_scale).reindex(base.index).interpolate()
            # compare to baseline AC
            validation = validate_against_measured(base["Pac_W_total"], meas)
        except Exception as e:
            validation = {"error": str(e)}

    summary = {
        "date": args.date, "lat": args.lat, "lon": args.lon,
        "mode": args.mode, "modules": args.modules,
        "E_Wh_tracking": res["E_Wh_tracking"],
        "E_Wh_opt":      res.get("E_Wh_opt"),
        "delta_Wh":      res.get("delta_Wh"),
        "delta_pct":     res.get("delta_pct"),
        "params": {
            "albedo": args.albedo, "p_stc": args.p_stc, "gamma": args.gamma,
            "noct": args.noct, "u0": args.u0, "u1": args.u1,
            "use_faiman": bool(args.faiman),
            "axis_azm": args.axis_azm, "axis_tilt": args.axis_tilt,
            "max_angle": args.max_angle, "backtrack": bool(args.backtrack), "gcr": args.gcr,
            "offset_range": [omin, omax, ostep], "tilt_bounds": [tmin, tmax],
            "eta_inverter": args.eta_inverter, "dc_wiring_loss": args.dc_wiring_loss
        },
        "validation": validation
    }
    with open(f"summary_{tag}.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary_{tag}.json")

    # Final log line
    elog = f"Baseline AC E_day={summary['E_Wh_tracking']:.1f} Wh"
    if summary["E_Wh_opt"] is not None:
        elog += f" | Adaptive AC E_day={summary['E_Wh_opt']:.1f} Wh"
        elog += f" | Δ={summary['delta_Wh']:.1f} Wh ({summary['delta_pct']:.2f}%)"
    print("OK |", args.date, args.mode, "|", elog)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
