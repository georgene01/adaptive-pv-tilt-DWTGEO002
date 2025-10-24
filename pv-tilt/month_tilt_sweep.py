import requests, pandas as pd, pvlib, matplotlib.pyplot as plt
from pvlib.location import Location

# --- Site & module (same as before) ---
lat, lon = -33.9249, 18.4241
azm_deg = 0
P_STC = 440.0
GAMMA = -0.0035
NOCT  = 46.0
TZ = 'Africa/Johannesburg'

def fetch_month(yyyymm):
    y, m = yyyymm.split("-")
    days_in_month = {"01":31,"02":28,"03":31,"04":30,"05":31,"06":30,
                     "07":31,"08":31,"09":30,"10":31,"11":30,"12":31}[m]
    start = f"{y}{m}01"
    end   = f"{y}{m}{days_in_month:02d}"

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": lat, "longitude": lon,
        "start": start, "end": end,
        "community": "RE",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "temporal": "hourly",          # explicit
        "time-standard": "UTC",        # explicit
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    par = r.json()["properties"]["parameter"]

    ghi_dict = par["ALLSKY_SFC_SW_DWN"]
    t2m_dict = par["T2M"]
    keys = sorted(ghi_dict.keys())                     # 'YYYYMMDDHH'
    idx_utc = pd.to_datetime(keys, format="%Y%m%d%H", utc=True)

    ghi = pd.Series([ghi_dict[k] for k in keys], index=idx_utc, dtype=float)
    t2m = pd.Series([t2m_dict[k] for k in keys], index=idx_utc, dtype=float)

    return pd.DataFrame({"GHI": ghi, "T2M": t2m})


def month_energy_vs_tilt(yyyymm):
    df = fetch_month(yyyymm)
    site = Location(lat, lon, TZ, 25, 'Cape Town')
    df = df.tz_convert(TZ)
    solpos = site.get_solarposition(df.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    erbs = pvlib.irradiance.erbs(df["GHI"], solpos["zenith"], df.index)
    dni, dhi = erbs["dni"], erbs["dhi"]

    tilts = range(0, 71, 5)
    E = {}
    for tilt in tilts:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt, surface_azimuth=azm_deg,
            dni=dni, ghi=df["GHI"], dhi=dhi, dni_extra=dni_extra,
            solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"],
            model="haydavies"
        )
        G_POA = poa["poa_global"].clip(lower=0)
        T_cell = df["T2M"] + (NOCT - 20.0)*(G_POA/800.0)
        Pdc = P_STC * (G_POA/1000.0) * (1.0 + GAMMA*(T_cell - 25.0))
        Pdc = Pdc.clip(lower=0)
        E[tilt] = Pdc.sum()  # Wh for the whole month (hourly data)
    best = max(E, key=E.get)
    return pd.Series(E), best

# --- Run for June and December 2025 ---
E_jun, best_jun = month_energy_vs_tilt("2025-06")
E_dec, best_dec = month_energy_vs_tilt("2025-12")

print(f"Best tilt June 2025  : {best_jun}°  (Energy {E_jun[best_jun]:.0f} Wh)")
print(f"Best tilt Dec  2025  : {best_dec}°  (Energy {E_dec[best_dec]:.0f} Wh)")

# Plot
plt.figure(figsize=(8,5))
plt.plot(E_jun.index, E_jun.values, marker="o", label="June 2025 (winter)")
plt.plot(E_dec.index, E_dec.values, marker="o", label="Dec  2025 (summer)")
plt.title("Monthly Energy vs Tilt (Cape Town)")
plt.xlabel("Tilt angle (deg)")
plt.ylabel("Energy (Wh per module, month total)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

