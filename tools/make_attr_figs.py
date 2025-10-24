# tools/make_attr_figs.py
"""
Generate attribution figures from:
  - tables/attribution_summary.csv   (per-day period shares + ΔTc)
  - tables/hourly_attribution_all.csv (per-hour ΔP info)

Outputs (written to figs/):
  - fig_attr_gain_by_period_stacked.pdf (and .png)
  - fig_attr_hourly_heatmap.pdf (and .png)

Usage (from repo root):
  python tools/make_attr_figs.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(".")
TABLES = ROOT / "tables"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

def load_inputs():
    attrib = pd.read_csv(TABLES / "attribution_summary.csv")
    hourly = pd.read_csv(TABLES / "hourly_attribution_all.csv")
    # Basic sanity
    for c in ["site","date","frac_morn","frac_mid","frac_eve"]:
        if c not in attrib.columns:
            raise RuntimeError(f"Column '{c}' missing in {TABLES/'attribution_summary.csv'}")
    for c in ["site","date","pac_base_W","pac_opt_W","delta_W","hour_local"]:
        if c not in hourly.columns:
            raise RuntimeError(f"Column '{c}' missing in {TABLES/'hourly_attribution_all.csv'}")
    return attrib, hourly

def fig_stacked_by_period(attrib: pd.DataFrame, out_pdf: Path, out_png: Path):
    # Order by site then date
    data = attrib.copy().sort_values(["site","date"]).reset_index(drop=True)
    labels = data["site"] + " " + data["date"]
    m = data["frac_morn"].to_numpy()
    d = data["frac_mid"].to_numpy()
    e = data["frac_eve"].to_numpy()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x, m, label="Morning (0–11h)")
    ax.bar(x, d, bottom=m, label="Midday (12–13h)")
    ax.bar(x, e, bottom=m+d, label="Evening (14–23h)")
    ax.set_ylabel("Fraction of daily gain (share of ΔE)")
    ax.set_title("Attribution of daily gain by period (per exemplar day)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl if i % 2 == 0 else "" for i, lbl in enumerate(labels)], rotation=90)
    ax.legend(loc="upper right", ncols=1)
    ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_pdf)

def _per_day_hour_share(df: pd.DataFrame) -> pd.Series:
    """Return length-24 vector: share of ΔE by hour for one (site,date) group."""
    # keep daylight (any power)
    g = df[(df["pac_base_W"]>0) | (df["pac_opt_W"]>0)].copy()
    s = g["delta_W"].sum()
    if s == 0 or np.isclose(s, 0.0):
        w = g["delta_W"] * 0.0
    else:
        w = g["delta_W"] / s
    out = w.groupby(g["hour_local"]).sum().reindex(range(24), fill_value=0.0)
    return out

def fig_hourly_heatmap(hourly: pd.DataFrame, attrib: pd.DataFrame, out_pdf: Path, out_png: Path):
    # Build per-day key, compute share by hour for each day
    hourly = hourly.copy()
    hourly["key"] = hourly["site"] + " " + hourly["date"]
    # Desired row order: site/date as in attribution_summary
    order = (attrib.sort_values(["site","date"])["site"] + " " + attrib["date"]).tolist()

    H_rows = []
    keys = []
    for key, g in hourly.groupby("key"):
        H_rows.append(_per_day_hour_share(g))
        keys.append(key)
    H = pd.DataFrame(H_rows, index=keys)  # rows=day, cols=0..23
    # Reindex rows to our preferred order (ignore missing with inner index)
    H = H.reindex(order)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    im = ax.imshow(H.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title("Hourly contribution share to daily gain (per day)")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Day (site date)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_yticks(np.arange(len(H.index)))
    # downsample row labels for readability
    ylabels = [lbl if i % 2 == 0 else "" for i, lbl in enumerate(H.index.tolist())]
    ax.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of ΔE")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_pdf)

def main():
    attrib, hourly = load_inputs()
    fig_stacked_by_period(
        attrib,
        FIGS / "fig_attr_gain_by_period_stacked.pdf",
        FIGS / "fig_attr_gain_by_period_stacked.png",
    )
    fig_hourly_heatmap(
        hourly,
        attrib,
        FIGS / "fig_attr_hourly_heatmap.pdf",
        FIGS / "fig_attr_hourly_heatmap.png",
    )

if __name__ == "__main__":
    main()
