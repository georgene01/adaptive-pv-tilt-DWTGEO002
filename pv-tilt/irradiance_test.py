import requests
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from pvlib.location import Location

# --- Config (you can tweak these later) ---
lat, lon = -33.9249, 18.4241            # Cape Town
tilt_deg, azm_deg = 30, 0               # north-facing in SA
P_STC = 440.0                           # W
GAMMA = -0.0035                         # 1/°C  (=-0.35%/°C)
NOCT  = 46.0                            # °C

# --- A) Fetch NASA POWER data for ONE day (GHI + T2M) ---
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
print("Fetching NASA POWER (1 day)...")
r = requests.get(url, params=params, timeout=60); r.raise_for_status()
par = r.json()["properties"]["parameter"]
ghi = pd.Series(par["ALLSKY_SFC_SW_DWN"], dtype=float)
t2m = pd.Series(par["T2M"], dtype=float)

# Make UTC index, then convert to local tz
idx_utc = pd.to_datetime(ghi.index, format="%Y%m%d%H", utc=True)
df = pd.DataFrame({"GHI": ghi.values, "T2M": t2m.values}, index=idx_utc)

# --- B) Solar geometry (local time) ---
site = Location(lat, lon, 'Africa/Johannesburg', 25, 'Cape Town')
df = df.tz_convert(site.tz)
solpos = site.get_solarposition(df.index)

# --- C) Decompose GHI -> DNI/DHI and transpose to POA ---
erbs = pvlib.irradiance.erbs(df["GHI"], solpos["zenith"], df.index)
dni, dhi = erbs["dni"], erbs["dhi"]
dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt_deg, surface_azimuth=azm_deg,
    dni=dni, ghi=df["GHI"], dhi=dhi, dni_extra=dni_extra,
    solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"],
    model="haydavies"
)
G_POA = poa["poa_global"].clip(lower=0)

# --- D) Cell temperature (simple NOCT) ---
# T_cell = T_amb + (NOCT - 20)*(G_POA/800)
T_cell = df["T2M"] + (NOCT - 20.0)*(G_POA/800.0)

# --- E) Module DC power with temp coefficient ---
# Pdc = P_STC * (G_POA/1000) * [1 + GAMMA*(T_cell - 25)]
Pdc = P_STC * (G_POA/1000.0) * (1.0 + GAMMA*(T_cell - 25.0))
Pdc = Pdc.clip(lower=0)

print(pd.DataFrame({"G_POA": G_POA, "T_cell": T_cell, "Pdc_W": Pdc}).head(10))

# --- F) Quick plot ---
plt.figure(figsize=(10,6))
G_POA.plot(label="G_POA (W/m²)")
T_cell.plot(label="T_cell (°C)")
Pdc.plot(label="Pdc (W)")
plt.title(f"Cape Town • {df.index[0].date()} • Tilt={tilt_deg}°")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

