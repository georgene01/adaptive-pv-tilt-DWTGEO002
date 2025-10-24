# tools/make_figs_overlay.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project helpers
from tools.sim_support import build_sim_config, run_daily

ROOT = Path(".")
TABLES = ROOT / "tables"
FIGS = ROOT / "figs"

SUMMARY_CSV = TABLES / "daily_energy_all.csv"          # already built
MANIFEST_CSV = ROOT / "selection" / "exemplars_manifest.csv"  # optional for captions

def _theta_star_from_summary(site, date):
    df = pd.read_csv(SUMMARY_CSV)
    row = df[(df["site"].str.upper()==site.upper()) & (df["date"]==date)]
    if row.empty:
        raise RuntimeError(f"No row in {SUMMARY_CSV} for {site} {date}")
    r = row.iloc[0]
    return float(r["offset_opt_deg"]), float(r["E_perp_kWh"]), float(r["E_opt_kWh"]), float(r["delta_kWh"]), float(r["delta_pct"]), str(r.get("bin","")), str(r.get("season",""))

def sweep_plot_with_annotation(site, date):
    """
    Read tables/Eday_sweep_<SITE>_<DATE>.csv, convert Wh->kWh, annotate θ★ + Δ%.
    Save figs/Eday_sweep_<SITE>_<DATE>_annot.pdf
    """
    sweep_path = TABLES / f"Eday_sweep_{site}_{date}.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(sweep_path)

    df = pd.read_csv(sweep_path)
    if not {"offset_deg","e_day_Wh"} <= set(df.columns):
        raise RuntimeError(f"{sweep_path} missing required columns.")
    df = df.copy()
    df["e_day_kWh"] = df["e_day_Wh"] / 1000.0

    # Get θ★ and Δ from summary
    theta_star, E0_kWh, Eopt_kWh, dE_kWh, d_pct, binlab, season = _theta_star_from_summary(site, date)

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["offset_deg"], df["e_day_kWh"], lw=1.8)
    # mark θ★ on curve
    if theta_star in df["offset_deg"].values:
        ystar = float(df.loc[df["offset_deg"]==theta_star, "e_day_kWh"].iloc[0])
    else:
        # fallback to numeric peak
        idx = df["e_day_kWh"].idxmax()
        theta_star = float(df.loc[idx,"offset_deg"])
        ystar = float(df.loc[idx,"e_day_kWh"])
    ax.scatter([theta_star],[ystar], s=45, zorder=5)

    # Labels / title
    ax.set_xlabel("Constant offset $\\delta$ [deg]")
    ax.set_ylabel("Daily energy [kWh]")
    title_bin = (binlab.split(":")[0] if isinstance(binlab,str) and ":" in binlab else binlab).strip()
    ax.set_title(f"{site} {date}  ({title_bin}, {season})")
    # Annotation
    ax.annotate(fr"$\theta^\star={theta_star:.0f}^\circ$, $\Delta={d_pct:.1f}\%$",
                xy=(theta_star, ystar),
                xytext=(theta_star+1, ystar*1.01),
                arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=10)

    ax.grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    out = FIGS / f"Eday_sweep_{site}_{date}_annot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)

def pac_overlay(site, date, base_dir="."):
    """
    Recompute AC power for δ=0° and δ=θ★ using your run_daily(),
    overlay both traces with legend, save to figs/Pac_<SITE>_<DATE>_overlay.pdf
    """
    theta_star, E0_kWh, Eopt_kWh, dE_kWh, d_pct, binlab, season = _theta_star_from_summary(site, date)
    sim = build_sim_config(site, base_dir=base_dir)

    # Resolve weather filename by site
    wmap = {"NC":"data_raw/NC_2024_POWER_qc.csv", "WC":"data_raw/WC_2024_POWER_qc.csv"}
    wpath = wmap.get(site.upper())
    if wpath is None or not Path(wpath).exists():
        # try a generic guess
        guess = list(Path("data_raw").glob(f"{site}_*qc*.csv"))
        if not guess:
            raise FileNotFoundError(f"Weather CSV not found for {site}.")
        wpath = str(guess[0])

    # Run both offsets
    out0 = run_daily(sim, wpath, date_local=date, offset_deg=0.0)
    outS = run_daily(sim, wpath, date_local=date, offset_deg=theta_star)

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(out0["time"], out0["pac"], label=r"$\delta=0^\circ$ (baseline)", lw=1.6)
    ax.plot(outS["time"], outS["pac"], label=fr"$\delta=\theta^\star={theta_star:.0f}^\circ$", lw=1.6)
    ax.set_ylabel("AC power [W]")
    ax.set_xlabel("Local time")
    title_bin = (binlab.split(":")[0] if isinstance(binlab,str) and ":" in binlab else binlab).strip()
    ax.set_title(f"{site} {date}  ({title_bin}, {season})")
    ax.grid(True, ls=":", lw=0.6)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = FIGS / f"Pac_{site}_{date}_overlay.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)

def make_for_examples(pairs):
    for site, date in pairs:
        sweep_plot_with_annotation(site, date)
        pac_overlay(site, date)

if __name__ == "__main__":
    # Default demo set — edit to taste
    examples = [
        ("WC","2024-03-02"),  # high gain
        ("NC","2024-08-29"),  # high gain
        ("NC","2024-03-21"),  # low gain
        ("WC","2024-10-23"),  # low gain
    ]
    make_for_examples(examples)

