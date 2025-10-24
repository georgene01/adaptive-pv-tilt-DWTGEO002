import requests, pandas as pd, pvlib, matplotlib.pyplot as plt
from pvlib.location import Location

# --- Site & module parameters ---
lat, lon = -33.9249, 18.4241   # Cape Town
azm_deg = 0                    # facing north
P_STC = 440.0                  # module rated power (W)
GAMMA = -0.0035                # temp coefficient (1/°C)
NOCT  = 46.0                   # Nominal Operating Cell Temp (°C)
TZ = 'Africa/Johannesburg'

def fetch_day(date_str="2025-01-01"):
    """Fetch 1 full day of NASA POWER data (hourly)."""
    y, m, d = date_str.split("-")
    start = f"{y}{m}{d}"
    end   = f"{y}{m}{d}"
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = dict(latitude=lat, longitude=lon, start=start, end=end,
                  community="RE", parameters="ALLSKY_SFC_SW_DWN,T2M",
                  format="JSON")
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    par = r.json()["properties"]["parameter"]
    ghi = pd.Series(par["ALLSKY_SFC_SW_DWN"], dtype=float)
    t2m = pd.Series(par["T2M"], dtype=float)
    idx_utc = pd.to_datetime(ghi.index, format="%Y%m%d%H", utc=True)
    df = pd.DataFrame({"GHI": ghi.values, "T2M": t2m.values}, index=idx_utc)
    return df.tz_convert(TZ)

def best_tilt_each_hour(date_str="2025-01-01"):
    df = fetch_day(date_str)
    site = Location(lat, lon, TZ, 25, 'Cape Town')
    solpos = site.get_solarposition(df.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    erbs = pvlib.irradiance.erbs(df["GHI"], solpos["zenith"], df.index)
    dni, dhi = erbs["dni"], erbs["dhi"]

    results = []
    tilts = range(0, 71, 2)  # finer grid (every 2°)
    for ts in df.index:
        if df.loc[ts, "GHI"] < 1:   # skip night hours
            results.append((ts.hour, None, 0))
            continue
        best_p = -1; best_t = None
        for tilt in tilts:
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt, surface_azimuth=azm_deg,
                dni=dni[ts], ghi=df["GHI"][ts], dhi=dhi[ts],
                dni_extra=dni_extra[ts],
                solar_zenith=solpos.loc[ts,"zenith"],
                solar_azimuth=solpos.loc[ts,"azimuth"],
                model="haydavies"
            )
            G_POA = max(0, poa["poa_global"])
            T_cell = df["T2M"][ts] + (NOCT - 20.0)*(G_POA/800.0)
            Pdc = P_STC * (G_POA/1000.0) * (1.0 + GAMMA*(T_cell - 25.0))
            if Pdc > best_p:
                best_p, best_t = Pdc, tilt
        results.append((ts.hour, best_t, best_p))
    return pd.DataFrame(results, columns=["hour","best_tilt","Pdc_W"])

# --- Run for 1 Jan 2025 ---
res = best_tilt_each_hour("2025-01-01")

print(res)

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(res["hour"], res["best_tilt"], marker="o")
plt.title("Thermal-aware Best Tilt vs Hour (Cape Town, 2025-01-01)")
plt.xlabel("Hour of Day")
plt.ylabel("Best Tilt (deg)")
plt.grid(True); plt.tight_layout()
plt.show()

