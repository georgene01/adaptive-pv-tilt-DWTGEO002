import requests
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from pvlib.location import Location

# --- Config ---
lat, lon = -33.9249, 18.4241
azm_deg = 0  # north-facing
P_STC = 440.0
GAMMA = -0.0035
NOCT  = 46.0

# --- A) Fetch NASA POWER data for ONE DAY (for demo) ---
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "latitude": lat,
    "longitude": lon,
    "start": "20250101",
    "end": "20250101",
    "community": "RE",
    "parameters": "ALLSKY_SFC_SW_DWN,T2M",
    "format": "JSON"
}
print("Fetching NASA POWER data...")
r = requests.get(url, params=params, timeout=60); r.raise_for_status()
par = r.json()["properties"]["parameter"]
ghi = pd.Series(par["ALLSKY_SFC_SW_DWN"], dtype=float)
t2m = pd.Series(par["T2M"], dtype=float)

idx_utc = pd.to_datetime(ghi.index, format="%Y%m%d%H", utc=True)
df = pd.DataFrame({"GHI": ghi.values, "T2M": t2m.values}, index=idx_utc)

# Localize
site = Location(lat, lon, 'Africa/Johannesburg', 25, 'Cape Town')
df = df.tz_convert(site.tz)
solpos = site.get_solarposition(df.index)
dni_extra = pvlib.irradiance.get_extra_radiation(df.index)

# --- B) Tilt sweep ---
tilts = range(0, 71, 5)  # 0, 5, 10 … 70 deg
energy_results = {}

for tilt in tilts:
    # Decompose irradiance
    erbs = pvlib.irradiance.erbs(df["GHI"], solpos["zenith"], df.index)
    dni, dhi = erbs["dni"], erbs["dhi"]

    # POA irradiance
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azm_deg,
        dni=dni, ghi=df["GHI"], dhi=dhi, dni_extra=dni_extra,
        solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"],
        model="haydavies"
    )
    G_POA = poa["poa_global"].clip(lower=0)

    # Cell temperature
    T_cell = df["T2M"] + (NOCT - 20.0)*(G_POA/800.0)

    # DC power
    Pdc = P_STC * (G_POA/1000.0) * (1.0 + GAMMA*(T_cell - 25.0))
    Pdc = Pdc.clip(lower=0)

    # Daily energy [Wh] = sum(Pdc * 1h)
    E_day = Pdc.sum()  # since hourly data, Wh
    energy_results[tilt] = E_day

# --- C) Plot Energy vs Tilt ---
tilt_list = list(energy_results.keys())
E_list = list(energy_results.values())

plt.figure(figsize=(8,5))
plt.plot(tilt_list, E_list, marker="o")
plt.title("Daily Energy vs Tilt (Cape Town, 1 Jan 2025)")
plt.xlabel("Tilt angle (deg)")
plt.ylabel("Energy (Wh per module)")
plt.grid(True)
plt.show()

print("Best tilt =", max(energy_results, key=energy_results.get))

