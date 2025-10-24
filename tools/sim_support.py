
"""
Simulation helpers for daily PV modelling with pvlib (with graceful fallbacks).

Features:
- Loads project configs (models/module/inverter/tracker)
- Builds a SimConfig with site metadata (lat/lon/tz)
- Loads QC'd weather (tz-aware local index)
- Decomposes GHI -> (DNI, DHI) using ERBS
- Computes POA on single-axis tracker (Hay-Davies) with constant offset δ
- Faiman module temperature
- DC (DeSoto) and AC (Sandia/CEC) power
- Daily energy integration using actual time deltas
- Offset sweep utility for δ in [min, max] with step

Requirements:
  pip install pvlib pandas pyyaml matplotlib numpy
"""
from __future__ import annotations
import pandas as pd


# --- injected helper for local runs ---
try:
    from types import SimpleNamespace
except Exception:
    pass

def build_simulator(site: str):
    """Return a minimal simulator object with fields used by run_step2_daily/poa_on_tracker.
    Adjust lat/lon/tz if needed; defaults are SA sites.
    """
    s = (site or "").upper()
    if s == "NC":
        meta = dict(lat=-28.46, lon=21.23, tz="Africa/Johannesburg")
    elif s == "WC":
        meta = dict(lat=-33.92, lon=18.42, tz="Africa/Johannesburg")
    else:
        # fallback: Johannesburg-ish
        meta = dict(lat=-26.20, lon=28.04, tz="Africa/Johannesburg")

    tracker = dict(
        axis_tilt_deg=0.0,
        axis_azimuth_deg=0.0,
        gcr=0.35,
        backtrack=True,
        max_rotation_deg=60.0,
    )
    module = dict(pdc0=1.0)  # placeholder so callers don't crash if they touch it

    # common extras some code paths read
    extras = dict(albedo=0.2, elevation_m=0.0)

    try:
        return SimpleNamespace(**meta, tracker=tracker, module=module, **extras)
    except NameError:
        # if SimpleNamespace import failed for some reason
        class S(dict):
            __getattr__ = dict.get
        o = S(**meta | {"tracker": tracker, "module": module} | extras)
        return o
# --- end injected helper ---


import numpy as np
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Try pvlib import early so we can fail fast with a clear message
try:
    import pvlib
    from pvlib.temperature import faiman
    from pvlib import location, irradiance, pvsystem, tracking
    PVLIB_AVAILABLE = True
except Exception as e:
    PVLIB_AVAILABLE = False
    PVLIB_IMPORT_ERROR = str(e)


# -----------------------------
# Data container for simulation
# -----------------------------
@dataclass
class SimConfig:
    site_name: str
    lat: float
    lon: float
    tz: str
    models: Dict[str, Any]
    module: Dict[str, Any]
    inverter: Dict[str, Any]
    tracker: Dict[str, Any]


# -----------------------------
# Config + site loading
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_project_configs(base_dir: str = ".") -> Dict[str, Any]:
    models = load_yaml(f"{base_dir}/configs/models.yaml")
    module = load_yaml(f"{base_dir}/configs/module_trina415.yaml")
    inverter = load_yaml(f"{base_dir}/configs/inverter_sg2500hv.yaml")
    tracker = load_yaml(f"{base_dir}/configs/tracker_sat.yaml")
    return {"models": models, "module": module, "inverter": inverter, "tracker": tracker}

def load_site_row(sites_csv: str, site: str) -> Dict[str, Any]:
    df = pd.read_csv(sites_csv)
    row = df.loc[df["site"].str.upper() == site.upper()]
    if row.empty:
        raise RuntimeError(f"Site '{site}' not found in {sites_csv}")
    r = row.iloc[0].to_dict()
    lat = float(r.get("lat") or r.get("latitude"))
    lon = float(r.get("lon") or r.get("longitude"))
    tz  = r.get("tz") or r.get("timezone") or "Africa/Johannesburg"
    return {"site": site.upper(), "lat": lat, "lon": lon, "tz": tz}

def build_sim_config(site: str, base_dir=".") -> SimConfig:
    if not PVLIB_AVAILABLE:
        raise RuntimeError(f"pvlib is required for simulation but could not be imported: {PVLIB_IMPORT_ERROR}")
    cfgs = load_project_configs(base_dir)
    site_row = load_site_row(f"{base_dir}/inputs/sites.csv", site)
    return SimConfig(
        site_name=site_row["site"],
        lat=site_row["lat"],
        lon=site_row["lon"],
        tz=site_row["tz"],
        models=cfgs["models"],
        module=cfgs["module"],
        inverter=cfgs["inverter"],
        tracker=cfgs["tracker"],
    )

def _ensure_pvlib():
    if not PVLIB_AVAILABLE:
        raise RuntimeError(f"pvlib is required for simulation but could not be imported: {PVLIB_IMPORT_ERROR}")


# -----------------------------
# Weather I/O & preprocessing
# -----------------------------def load_weather_qc(path_csv: str, tz_name: str):
    """
    Build/verify a 'time_utc' column from several possible inputs, then return df.
    Accepted inputs (first match wins):
      1) 'time_utc' (parse as UTC)
      2) 'time_local' (localize/convert using tz_name)
      3) POWER columns YEAR/(MO|MONTH)/(DY|DA|DAY)/(HR|HOUR)
      4) 'time' (assume local -> tz_name -> UTC)
      5) Filename "YYYY-MM-DD" + HR/HOUR column or 24-row heuristic
    """
    import os, re
    import pandas as pd
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    df = pd.read_csv(path_csv, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    # 1) Already UTC
    if "time_utc" in cols:
        col = cols["time_utc"]
        ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        if ts.isna().any():
            bad = df.loc[ts.isna(), col].head().tolist()
            raise ValueError(f"time_utc present but unparseable: {bad}")
        df["time_utc"] = ts
        return df

    # 2) Local time → UTC
    if "time_local" in cols:
        col = cols["time_local"]
        ts = pd.to_datetime(df[col], errors="coerce")
        if ts.isna().any():
            bad = df.loc[ts.isna(), col].head().tolist()
            raise ValueError(f"time_local unparseable: {bad}")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)
        df["time_utc"] = ts.dt.tz_convert("UTC")
        return df

    # 3) POWER pieces: YEAR/(MO|MONTH)/(DY|DA|DAY)/(HR|HOUR)
    year = cols.get("year")
    mo   = cols.get("mo") or cols.get("month")
    dy   = cols.get("dy") or cols.get("da") or cols.get("day")
    hr   = cols.get("hr") or cols.get("hour")
    if year and mo and dy and hr:
        Y = pd.to_numeric(df[year], errors="coerce")
        M = pd.to_numeric(df[mo],   errors="coerce")
        D = pd.to_numeric(df[dy],   errors="coerce")
        H = pd.to_numeric(df[hr],   errors="coerce")
        if any(x.isna().any() for x in (Y, M, D, H)):
            raise ValueError("YEAR/MO/DY/HR present but contain non-numeric values.")
        ts = pd.to_datetime(
            Y.astype(int).astype(str) + "-" +
            M.astype(int).astype(str).str.zfill(2) + "-" +
            D.astype(int).astype(str).str.zfill(2) + " " +
            H.astype(int).astype(str).str.zfill(2) + ":00:00",
            errors="coerce"
        )
        if ts.isna().any():
            raise ValueError("Failed to construct timestamps from YEAR/MO/DY/HR.")
        ts = ts.dt.tz_localize(tz).dt.tz_convert("UTC")
        df["time_utc"] = ts
        return df

    # 4) Generic 'time' (assume local)
    if "time" in cols:
        col = cols["time"]
        ts  = pd.to_datetime(df[col], errors="coerce")
        if ts.isna().any():
            bad = df.loc[ts.isna(), col].head().tolist()
            raise ValueError(f"time unparseable: {bad}")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)
        df["time_utc"] = ts.dt.tz_convert("UTC")
        return df

    # 5) Filename-derived YYYY-MM-DD + hour column (or 24-row heuristic)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path_csv))
    if m:
        date_str = m.group(1)
        hcol = cols.get("hr") or cols.get("hour")
        if hcol:
            H = pd.to_numeric(df[hcol], errors="coerce").fillna(0).astype(int).clip(0, 23)
        else:
            import numpy as np
            if len(df) == 24:
                H = pd.Series(np.arange(24), index=df.index)
            else:
                raise ValueError("No HR/HOUR column and not 24 rows; cannot infer hours.")
        base = pd.to_datetime(date_str + " 00:00:00").tz_localize(tz)
        ts   = (base + pd.to_timedelta(H, unit="h")).tz_convert("UTC")
        df["time_utc"] = ts
        return df

    raise ValueError("Could not construct time_utc. Tried time_utc, time_local, YEAR/MO/DY/HR, time, filename+hour.")

def erbs_decompose(df_local: pd.DataFrame, loc: "pvlib.location.Location") -> pd.DataFrame:
    """
    Decompose GHI->DNI/DHI using ERBS with solar position.
    Adds columns: dni_wm2, dhi_wm2; applies night mask (zenith >= 90 deg).
    """
    _ensure_pvlib()

    # Solar position (tz-aware index is required)
    solpos = loc.get_solarposition(df_local.index)
    zen = solpos["apparent_zenith"].clip(upper=90)

    ghi = df_local["ghi_wm2"].clip(lower=0).astype(float)

    # ERBS decomposition (works well for hourly data)
    erbs = irradiance.erbs(ghi, zen, df_local.index)
    dni = erbs["dni"].fillna(0.0).clip(lower=0.0)
    dhi = erbs["dhi"].fillna(0.0).clip(lower=0.0)

    # Night cleanup
    night = zen >= 90
    dni = dni.where(~night, 0.0)
    dhi = dhi.where(~night, 0.0)

    out = df_local.copy()
    out["dni_wm2"] = dni
    out["dhi_wm2"] = dhi
    return out


# -----------------------------
# POA on single-axis tracker
# -----------------------------
def poa_on_tracker(df: pd.DataFrame, sim: SimConfig, backtrack: bool = True, offset_deg: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute POA components for a single-axis tracker with a constant tilt offset δ.
    Uses Hay–Davies transposition and provides dni_extra (required).
    Returns: (poa_df, tracker_df)
    """
    _ensure_pvlib()

    # Location / solar position
    loc = location.Location(sim.lat, sim.lon, tz=sim.tz)
    solpos = loc.get_solarposition(df.index)
    zen = solpos["apparent_zenith"].clip(upper=90)

    # Tracker geometry (from configs)
    axis_tilt = float(sim.tracker.get("axis_tilt_deg", 0.0))
    axis_azimuth = float(sim.tracker.get("axis_azimuth_deg", 0.0))
    max_angle = float(sim.tracker.get("max_angle_deg", 60.0))
    gcr = float(sim.tracker.get("gcr", 0.30))

    trk = tracking.singleaxis(
        apparent_zenith=zen,
        apparent_azimuth=solpos["azimuth"],
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack,
        gcr=gcr
    )

    # Apply constant offset δ to the surface tilt (clip to [0, max_angle])
    trk["surface_tilt"] = (trk["surface_tilt"] + float(offset_deg)).clip(lower=0, upper=max_angle)

    # Required extraterrestrial DNI for Hay–Davies
    dni_extra = irradiance.get_extra_radiation(df.index, method="spencer")

    # Transposition (Hay–Davies)
    poa = irradiance.get_total_irradiance(
        surface_tilt=trk["surface_tilt"],
        surface_azimuth=trk["surface_azimuth"],
        dni=df["dni_wm2"],
        ghi=df["ghi_wm2"],
        dhi=df["dhi_wm2"],
        solar_zenith=zen,
        solar_azimuth=solpos["azimuth"],
        dni_extra=dni_extra,         # <-- this fixes the previous error
        albedo=df.get("albedo", 0.2),
        model="haydavies"
    )
    return poa, trk


# -----------------------------
# Temperature & Power chains
# -----------------------------
def faiman_cell_temp(poa_irr: pd.Series, temp_air: pd.Series, wind: pd.Series, u0=25.0, u1=6.0) -> pd.Series:
    """
    Faiman model for module temperature (cell/backsheet proxy).
    u0 [W/m^2/K], u1 [W/(m^2*K)/(m/s)].
    """
    return faiman(poa_global=poa_irr, temp_air=temp_air, wind_speed=wind, u0=u0, u1=u1)

def dc_ac_power(sim: SimConfig, poa: pd.DataFrame, tcell: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute DC (De Soto single-diode) and inverter AC.
    - Uses cec/sapm-like dicts from configs where available.
    - Falls back gracefully if some coefficients are missing.
    Returns: (dc_df with p_dc & p_mp_raw, pac series)
    """
    _ensure_pvlib()

    # Module params (prefer CEC params; otherwise accept SAPM)
    mod = sim.module.get("cec_params", {}) or sim.module.get("sapm_params", {})
    gamma_p = float(sim.module.get("gamma_p", -0.0034))  # per °C for Pmp

    # Effective irradiance using SAPM formulation if possible; fallback to POA global
    try:
        ee = pvsystem.sapm_effective_irradiance(
            poa_direct=poa["poa_direct"],
            poa_diffuse=poa["poa_diffuse"],
            airmass_absolute=None,
            airmass_relative=None,
            aoi=poa.get("aoi", None),
            module=mod
        )
        ee = ee.fillna(poa["poa_global"])
    except Exception:
        ee = poa["poa_global"]

    # De Soto single-diode parameters
    il, io, rs, rsh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance=ee,
        temp_cell=tcell,
        alpha_sc=mod.get("alpha_sc", 0.0045),
        a_ref=mod.get("a_ref", 1.3),
        I_L_ref=mod.get("I_L_ref", 9.5),
        I_o_ref=mod.get("I_o_ref", 1e-10),
        R_sh_ref=mod.get("R_sh_ref", 200.0),
        R_s=mod.get("R_s", 0.5),
        EgRef=1.121,
        dEgdT=-0.0002677
    )
    sd = pvsystem.singlediode(il, io, rs, rsh, nNsVth)
    p_dc = sd["p_mp"]  # W per module (approx)

    # Temperature correction (gamma_p relative to 25°C)
    delta_T = tcell - 25.0
    p_dc_corr = p_dc * (1.0 + gamma_p * delta_T)

    # Inverter model (Sandia/CEC); fallback to simple clipping
    # robust inverter access (default to nominal if missing)
    try:
        inv = sim.inverter.get("coefficients", {})
    except Exception:
        inv = {}
    if not inv:
        inv = {"eta_nom": 0.97}
    pac_nameplate = float(sim.inverter.get("pac_nameplate_w", 2_500_000.0))
    if inv:
        try:
            ac = pvsystem.snlinverter(p_dc_corr, v_dc=None, inverter=inv)
            pac = ac.clip(lower=0).clip(upper=pac_nameplate)
        except Exception:
            pac = p_dc_corr.clip(lower=0).clip(upper=pac_nameplate)
    else:
        pac = p_dc_corr.clip(lower=0).clip(upper=pac_nameplate)

    return pd.DataFrame({"p_dc": p_dc_corr, "p_mp_raw": p_dc}), pac


# -----------------------------
# Daily simulation & sweeping
# -----------------------------
def _integrate_energy_Wh(power_W: pd.Series) -> float:
    """
    Integrate power(t) over the index using actual time deltas to get energy in Wh.
    Handles irregular sampling; assumes tz-aware datetime index.
    """
    if power_W.empty:
        return 0.0
    s = power_W.copy()
    # dt to next sample in hours; last sample gets median dt to avoid drop
    idx = s.index
    dt_next = (pd.Series(idx[1:].append(idx[-1:])).reset_index(drop=True) - pd.Series(idx).reset_index(drop=True))
    dt_next = dt_next.dt.total_seconds().astype(float) / 3600.0
    if len(dt_next) > 1:
        dt_next.iloc[-1] = float(np.nanmedian(dt_next[:-1]))
    else:
        dt_next.iloc[-1] = 1.0  # assume hourly if single point
    dt_h = pd.Series(dt_next.values, index=s.index)
    energy_Wh = float((s.clip(lower=0) * dt_h).sum())
    return energy_Wh

def run_daily(sim: SimConfig, weather_csv: str, date_local: str, offset_deg: float) -> Dict[str, Any]:
    """
    Run the full stack for one local date and return dict with series & summaries.
    Steps:
      - load local tz weather
      - slice the day
      - decompose (ERBS)
      - tracker + POA (Hay–Davies with dni_extra)
      - Faiman temperature
      - DC/AC power
      - integrate to Wh using actual dt
    """
    _ensure_pvlib()

    loc = location.Location(sim.lat, sim.lon, tz=sim.tz)
    df_loc = load_weather_qc(weather_csv, sim.tz)

    # Select the 24h window for date_local (local midnight to next midnight)
    target_date = pd.to_datetime(date_local).date()
    day_mask = df_loc.index.date == target_date
    day = df_loc.loc[day_mask].copy()
    if day.empty:
        raise RuntimeError(f"No rows found for {date_local} in {weather_csv}")

    # Decompose + POA on tracker
    dfd = erbs_decompose(day, loc)
    poa, trk = poa_on_tracker(dfd, sim, backtrack=True, offset_deg=float(offset_deg))

    # Faiman (from configs)
    try:
        u0 = sim.models["temperature"]["baseline"]["parameters"]["u0_W_m2K"]
        u1 = sim.models["temperature"]["baseline"]["parameters"]["u1_W_s_m3K"]
    except Exception:
        # Sensible defaults if not present
        u0, u1 = 25.0, 6.0
    tc = faiman_cell_temp(poa["poa_global"], dfd["temp_air_c"], dfd["wind_speed_ms"], u0, u1)

    # DC/AC
    dc_df, pac = dc_ac_power(sim, poa, tc)

    # Daily energy (Wh)
    e_day_Wh = _integrate_energy_Wh(pac)

    return {
        "time": day.index,
        "poa": poa,
        "tc": tc,
        "pac": pac,
        "dc": dc_df,
        "trk": trk,
        "e_day_wh": float(e_day_Wh)
    }

def sweep_offsets(sim: SimConfig, weather_csv: str, date_local: str, min_deg: int = -30, max_deg: int = 30, step_deg: int = 2) -> pd.DataFrame:
    """
    Compute E_day(δ) sweep for a given date. Returns a DataFrame with columns:
      offset_deg, e_day_Wh
    """
    rows = []
    for d in range(int(min_deg), int(max_deg) + 1, int(step_deg)):
        try:
            out = run_daily(sim, weather_csv, date_local, offset_deg=float(d))
            rows.append({"offset_deg": d, "e_day_Wh": out["e_day_wh"]})
        except Exception:
            rows.append({"offset_deg": d, "e_day_Wh": np.nan})
    return pd.DataFrame(rows)


# --- patched defaults injection ---
def _local_build_simulator(site: str):
    from types import SimpleNamespace as S
    # Minimal, safe defaults for a single-axis tracker plant
    meta = dict(
        site=site,
        lat=-33.0,            # any valid lat/lon; not critical for code-path sanity
        lon=18.0,
        tz="Africa/Johannesburg",
    )
    tracker = dict(
        axis_tilt_deg=0.0,
        axis_azimuth_deg=0.0,
        max_angle_deg=60.0,
        gcr=0.35,
    )
    module = dict(
        pdc_stc_w=400.0,
        gamma_pdc_perC=-0.0035,
        noct_cell_C=45.0,
        cells_in_series=144,
        rho_ground=0.2,
    )
    inverter = dict(
        # keep it simple: pass-through with a nominal efficiency
        coefficients=dict(eta_nom=0.97)
    )
    from types import SimpleNamespace as S
    o = S(**(meta | {"tracker": tracker, "module": module, "inverter": inverter}))
    return o
# --- end patched defaults injection ---
