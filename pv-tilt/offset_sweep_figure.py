import os, subprocess, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- CONFIG ----------
DATE = "2025-01-15"          # choose the day
DELTA_MIN, DELTA_MAX, STEP = -15, 15, 1
PIPELINE = "pipeline_best.py"  # path if not in CWD
OUTDIR = os.path.join("outputs", DATE)
RUN_FLAGS = [
    "--mode", "singleaxis",
    "--axis-azm", "0",
    "--axis-tilt", "0",
    "--max-angle", "60",
    "--backtrack",
    "--gcr", "0.30",
    "--tilt-bounds", "0,60",
    "--faiman",
]

# Ensure folders
os.makedirs("figures", exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

def run_pipeline_for_delta(delta):
    """Calls your pipeline once for a single offset and expects it to write summary + hourly CSV."""
    # Typical pattern: pass a single delta via --offset-single so pipeline runs that one
    # If you only have --offset-range, you can still run once for the whole range and then parse.
    cmd = [
        sys.executable, PIPELINE,
        "--date", DATE,
        *RUN_FLAGS,
        "--offset-single", str(delta)     # <--- add this flag to your pipeline (see Option B if needed)
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def read_energy_from_summary(delta):
    """Read E_day_kWh from outputs/<DATE>/summary_delta_<±NN>.csv"""
    fname = os.path.join(OUTDIR, f"summary_delta_{int(delta):+d}.csv")  # e.g. summary_delta_+3.csv
    with open(fname, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            return float(row["E_day_kWh"])
    raise FileNotFoundError(fname)

def read_hourly_pac(delta):
    """Read hourly Pac series for plotting right panel."""
    fname = os.path.join(OUTDIR, f"pac_hourly_delta_{int(delta):+d}.csv")
    times, pac = [], []
    with open(fname, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            times.append(row["timestamp_local"])
            pac.append(float(row["P_ac_kW"]))
    # Convert to hour-of-day numbers (for nice x-axis)
    hours = [datetime.fromisoformat(t).hour + datetime.fromisoformat(t).minute/60 for t in times]
    return np.array(hours), np.array(pac)

def main():
    deltas = np.arange(DELTA_MIN, DELTA_MAX + STEP, STEP)

    # 1) run the pipeline for each delta
    for d in deltas:
        run_pipeline_for_delta(d)

    # 2) collect daily energies
    E_day = [read_energy_from_summary(d) for d in deltas]

    # 3) find best delta
    i_best = int(np.argmax(E_day))
    delta_star = int(deltas[i_best])
    print("Best delta:", delta_star, "deg")

    # 4) read time series for baseline and optimal
    h0, pac0 = read_hourly_pac(0)
    h1, pac1 = read_hourly_pac(delta_star)

    # 5) plot figure
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Left: sweep
    axes[0].plot(deltas, E_day, marker="o")
    axes[0].axvline(delta_star, linestyle="--", label=f"δ*={delta_star}°")
    axes[0].set_xlabel("Tilt offset δ [deg]")
    axes[0].set_ylabel("Daily AC Energy [kWh]")
    axes[0].set_title("Offset sweep: Energy vs δ")
    axes[0].legend()

    # Right: time series
    axes[1].plot(h0, pac0, label="δ=0° (baseline)")
    axes[1].plot(h1, pac1, label=f"δ*={delta_star}° (optimal)")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("AC Power [kW]")
    axes[1].set_title("Power curves: baseline vs optimal")
    axes[1].legend()

    plt.tight_layout()
    fn = "figures/offset_sweep_day.pdf"
    plt.savefig(fn)
    print("Saved:", os.path.abspath(fn))

if __name__ == "__main__":
    main()
