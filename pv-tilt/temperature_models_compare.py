import numpy as np
import matplotlib.pyplot as plt

# Example hourly data for a summer day (replace with POWER site data)
hours = np.arange(6, 19, 1)
Gpoa = np.array([0, 200, 500, 700, 900, 1000, 950, 800, 600, 400, 200, 50, 0])  # W/m²
Ta   = np.array([20, 22, 25, 28, 32, 35, 36, 35, 33, 30, 27, 24, 22])           # °C
u    = np.array([1, 1, 2, 3, 3, 4, 3, 2, 2, 2, 1, 1, 1])                        # m/s

# Parameters
T_NOCT = 45  # °C, from Trina datasheet
U0, U1 = 29, 6  # W/m²K, Faiman defaults

# --- Sandia (SAPM) model ---
a, b = -3.47, -0.0594   # SAPM open-rack, glass/backsheet defaults
Tmod_sandia = Ta + Gpoa * np.exp(a + b * u)    # module temp
Tc_sandia   = Tmod_sandia + 0.000267 * Gpoa    # cell temp

# --- NOCT model ---
Tc_noct   = Ta + (Gpoa/800.0) * (T_NOCT - 20)

# --- Faiman model ---
Tc_faiman = Ta + Gpoa / (U0 + U1*u)

# Plot
plt.figure(figsize=(8,5))
plt.plot(hours, Tc_noct, 'r-', label="NOCT model")
plt.plot(hours, Tc_sandia, 'b--', label="Sandia (SAPM) model")
plt.plot(hours, Tc_faiman, 'g-.', label="Faiman model")
plt.xlabel("Hour of day")
plt.ylabel("Cell temperature $T_c$ [°C]")
plt.title("Temperature model comparison (clear summer day, Northern Cape)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/temperature_models_compare.png", dpi=300)
plt.show()
