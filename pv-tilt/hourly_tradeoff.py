import requests, pandas as pd, numpy as np, matplotlib.pyplot as plt
import pvlib
from pvlib.location import Location

# ---- CONFIG (edit these three) ---------------------------------
LAT, LON = -33.9249, 18.4241          # Cape Town
DATE     = "2025-01-01"               # ISO date (local calendar day)
HOUR_LCL = 12                          # local hour to evaluate (0..23)
USE_CLEARSKY_IF_API_FAILS = True
# Module / thermal params (generic mono)
P_STC = 440.0          # W
GAMMA = -0.0035        # 1/°C  (=-0.35%/°C)
NOCT  = 46.0           # °C
AZM   = 0              # north-facing in SA
TZ    = "Africa/Johannesburg"
TILTS = range(0, 71, 2)
# ---------------------------------------------------------------

site = Location(LAT, LON, TZ, 25, "Site")
times = pd.date_range(DATE, periods=24, freq="1h", tz=TZ)    # local day (hourly)
solpos = site.get_solarposition(times)
dni_extra = pvlib.irradiance.get_extra_radiation(times)

def fetch_power_day(lat, lon, date_iso):
    ymd = date_iso.replace("-","")
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": lat, "longitude": lon,
        "start": ymd, "end": ymd,
        "community": "RE",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "temporal": "hourly",
        "time-standard": "UTC",
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    par = r.json()["properties"]["parameter"]
    ghi_dict, t2m_dict = par["ALLSKY_SFC_SW_DWN"], par["T2M"]
    keys = sorted(ghi_dict.keys())                         # 'YYYYMMDDHH'
    idx_utc = pd.to_datetime(keys, format="%Y%m%d%H", utc=True)
    df = pd.DataFrame({
        "GHI": [ghi_dict[k] for k in keys],
        "T2M": [t2m_dict[k] for k in keys],
    }, index=idx_utc).tz_convert(TZ)
    # reindex to our local hourly stamps (fill missing with 0/nearest)
    df = df.reindex(times).fillna(method="nearest").fillna(0.0)
    return df

try:
    df = fetch_power_day(LAT, LON, DATE)
    got_api = True
except Exception as e:
    if not USE_CLEARSKY_IF_API_FAILS:
        raise
    got_api = False
    # Clear‑sky fallback
    cs = site.get_clearsky(times)  # ghi/dni/dhi
    df = pd.DataFrame({"GHI": cs["ghi"], "T2M": 25.0}, index=times)  # assume 25 °C

# Decompose (GHI -> DNI/DHI) using Erbs
erbs = pvlib.irradiance.erbs(df["GHI"].clip(lower=0), solpos["zenith"], times)
dni, dhi = erbs["dni"].clip(lower=0), erbs["dhi"].clip(lower=0)

# Pick the target hour row
t0 = times[HOUR_LCL]
ghi0 = float(df.loc[t0, "GHI"])
t2m0 = float(df.loc[t0, "T2M"])
dni0 = float(dni.loc[t0])
dhi0 = float(dhi.loc[t0])
zen0 = float(solpos.loc[t0, "zenith"])
azz0 = float(solpos.loc[t0, "azimuth"])
dni_extra0 = float(dni_extra.loc[t0])

# Sweep tilts at that same moment
poa_list, tcell_list, pdc_list = [], [], []
for tilt in TILTS:
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=AZM,
        dni=dni0, ghi=ghi0, dhi=dhi0, dni_extra=dni_extra0,
        solar_zenith=zen0, solar_azimuth=azz0, model="haydavies"
    )
    g_poa = max(0.0, float(poa["poa_global"]))
    t_cell = t2m0 + (NOCT - 20.0)*(g_poa/800.0)
    pdc = max(0.0, P_STC*(g_poa/1000.0)*(1.0 + GAMMA*(t_cell - 25.0)))
    poa_list.append(g_poa); tcell_list.append(t_cell); pdc_list.append(pdc)

# Plot Pdc vs tilt (with POA and Tcell for insight)
best_idx = int(np.argmax(pdc_list))
best_tilt = list(TILTS)[best_idx]
print(f"Mode: {'NASA POWER' if got_api else 'Clear-sky'} • Date {DATE} • Hour {HOUR_LCL}:00")
print(f"Best tilt at that hour (thermal-aware) = {best_tilt}°  | Pdc ≈ {pdc_list[best_idx]:.1f} W")

fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(list(TILTS), pdc_list, marker="o", label="Pdc (W)")
ax1.set_xlabel("Tilt (deg)"); ax1.set_ylabel("Pdc (W)")
ax1.grid(True, alpha=0.3)
ax1.axvline(best_tilt, ls="--", lw=1)
ax2 = ax1.twinx()
ax2.plot(list(TILTS), poa_list, label="G_POA (W/m²)")
ax2.plot(list(TILTS), tcell_list, label="T_cell (°C)")
ax2.set_ylabel("G_POA / T_cell")
lines = ax1.get_lines()+ax2.get_lines()
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="best")
plt.title(f"{'NASA' if got_api else 'Clear-sky'} • {DATE} {HOUR_LCL}:00 • Cape Town")
plt.tight_layout(); plt.show()
