import requests
import pandas as pd

# Cape Town coordinates
lat, lon = -33.9249, 18.4241

# Request NASA POWER hourly data for ONE day (2025-01-01)
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "latitude": lat,
    "longitude": lon,
    "start": "20250101",
    "end": "20250101",
    "community": "RE",
    "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS2M,RH2M",
    "format": "JSON"
}

print("Fetching NASA POWER data...")
r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
js = r.json()

# Extract values
records = js["properties"]["parameter"]
times = sorted(list(records["ALLSKY_SFC_SW_DWN"].keys()))

rows = []
for t in times:
    rows.append({
        "time": pd.to_datetime(t, format="%Y%m%d%H", utc=True),
        "GHI": records["ALLSKY_SFC_SW_DWN"][t],  # W/m²
        "T2M": records["T2M"][t],                # °C
        "WS2M": records["WS2M"][t],              # m/s
        "RH2M": records["RH2M"][t]               # %
    })

df = pd.DataFrame(rows).set_index("time")

print(df.head())

