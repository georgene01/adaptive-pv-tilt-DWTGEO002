import pandas as pd, numpy as np, matplotlib.pyplot as plt, pvlib
from pvlib.location import Location

# --- Site, module, date ---
lat, lon, TZ = -33.9249, 18.4241, "Africa/Johannesburg"
P_STC, GAMMA, NOCT = 440.0, -0.0035, 46.0
date = "2025-01-01"

site = Location(lat, lon, TZ, 25, "Cape Town")
times = pd.date_range(date, freq="1h", periods=24, tz=TZ)
solpos = site.get_solarposition(times)
cs = site.get_clearsky(times)   # clear-sky baseline
dni, ghi, dhi = cs["dni"], cs["ghi"], cs["dhi"]
dni_extra = pvlib.irradiance.get_extra_radiation(times)

rows = []
for t in times:
    z, a = solpos.loc[t, "zenith"], solpos.loc[t, "azimuth"]
    if z > 90 or ghi.loc[t] < 1:
        rows.append((t.hour, np.nan, np.nan, np.nan, np.nan))
        continue

    # Standard tracker: perpendicular (tilt = zenith, azimuth = sun az)
    tilt_std, azm_std = z, a
    poa_std = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_std, surface_azimuth=azm_std,
        dni=dni.loc[t], ghi=ghi.loc[t], dhi=dhi.loc[t], dni_extra=dni_extra.loc[t],
        solar_zenith=z, solar_azimuth=a
    )["poa_global"]
    T_std = 25 + (NOCT - 20)*(poa_std/800.0)
    P_std = P_STC*(poa_std/1000.0)*(1 + GAMMA*(T_std - 25))

    # Thermal-aware: sweep ±20° around perpendicular (clamped 0..90)
    lo, hi = max(0, tilt_std-20), min(90, tilt_std+20)
    tilts = np.arange(lo, hi+0.1, 1)
    bestP, bestTilt = -1, tilt_std
    for tilt in tilts:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt, surface_azimuth=azm_std,
            dni=dni.loc[t], ghi=ghi.loc[t], dhi=dhi.loc[t], dni_extra=dni_extra.loc[t],
            solar_zenith=z, solar_azimuth=a
        )["poa_global"]
        T = 25 + (NOCT - 20)*(poa/800.0)
        P = P_STC*(poa/1000.0)*(1 + GAMMA*(T - 25))
        if P > bestP:
            bestP, bestTilt = P, tilt

    rows.append((t.hour, tilt_std, P_std, bestTilt, bestP))

df = pd.DataFrame(rows, columns=["hour","tilt_std","Pdc_std","tilt_opt","Pdc_opt"])

# --- Plot 1: Power vs hour ---
plt.figure(figsize=(9,5))
plt.plot(df["hour"], df["Pdc_std"], "k--o", label="Standard (perpendicular)")
plt.plot(df["hour"], df["Pdc_opt"], "r-o",  label="Thermal-aware optimum")
plt.xlabel("Hour of Day"); plt.ylabel("Power (W per module)")
plt.title(f"Thermal-aware vs Standard Tracking • {date}")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

# --- Plot 2: Tilt vs hour (and delta) ---
plt.figure(figsize=(9,5))
plt.plot(df["hour"], df["tilt_std"], "k--o", label="Standard tilt (zenith)")
plt.plot(df["hour"], df["tilt_opt"], "r-o",  label="Thermal-aware tilt")
plt.xlabel("Hour of Day"); plt.ylabel("Tilt (degrees)")
plt.title(f"Tilt: Standard vs Thermal-aware • {date}")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

# Optional: print delta tilt summary
delta = (df["tilt_opt"] - df["tilt_std"]).dropna()
print("Mean Δtilt (opt - std): {:.2f}° | Max Δtilt: {:.2f}°".format(delta.mean(), delta.abs().max()))
