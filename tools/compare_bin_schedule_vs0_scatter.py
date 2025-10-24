#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

SITES = ["NC", "WC"]
BINS  = ["clear", "mostly_clear", "partly_cloudy", "cloudy"]

BIN_STYLE = {
    "clear":         dict(color="#1f77b4", marker="o", label="clear"),
    "mostly_clear":  dict(color="#2ca02c", marker="^", label="mostly_clear"),
    "partly_cloudy": dict(color="#ff7f0e", marker="s", label="partly_cloudy"),
    "cloudy":        dict(color="#7f7f7f", marker="D", label="cloudy"),
}

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---------- Loaders ----------

def load_bin_curves(site: str) -> dict:
    """
    Returns {bin -> DataFrame with columns ['deg','gain_pct']} built from
    tables/offtilt_per_bin_{SITE}.csv (your sweep outputs).
    """
    f = Path(f"tables/offtilt_per_bin_{site}.csv")
    if not f.exists():
        raise SystemExit(f"[FATAL] Missing {f} — need per-bin sweep curves.")
    df = pd.read_csv(f)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # Expected columns (case-insensitive): bin_id, deg, mean_gain_pct_vs0
    if "mean_gain_pct_vs0" not in df.columns:
        if "gain_pct" in df.columns:
            df["mean_gain_pct_vs0"] = df["gain_pct"]
        else:
            raise SystemExit(f"[FATAL] {f} must contain 'mean_gain_pct_vs0'.")
    need = {"bin_id", "deg", "mean_gain_pct_vs0"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[FATAL] {f} must contain {need}; got {set(df.columns)}")

    curves = {}
    for b, g in df.groupby("bin_id"):
        gg = g.sort_values("deg")[["deg", "mean_gain_pct_vs0"]].copy()
        gg = gg.rename(columns={"mean_gain_pct_vs0": "gain_pct"})
        gg["deg"] = gg["deg"].astype(float)
        curves[str(b)] = gg.reset_index(drop=True)
    # Ensure all bins present
    for b in BINS:
        if b not in curves:
            # conservative fallback
            curves[b] = pd.DataFrame({"deg": [0.0], "gain_pct": [0.0]})
    return curves

def load_winners_angles(site: str) -> pd.Series:
    """
    Returns Series {bin -> theta_hat} (degrees).
    Prefer tables/offtilt_winners_{SITE}_compact.csv; otherwise derive from curves.
    """
    f = Path(f"tables/offtilt_winners_{site}_compact.csv")
    if f.exists():
        w = pd.read_csv(f).rename(columns={c: c.strip().lower() for c in pd.read_csv(f, nrows=0).columns})
        if not {"bin_id", "theta_hat"}.issubset(w.columns):
            raise SystemExit(f"[FATAL] {f} missing columns 'bin_id','theta_hat'.")
        s = w.set_index("bin_id")["theta_hat"].astype(float)
        # Ensure all bins
        for b in BINS:
            if b not in s.index:
                s.loc[b] = 0.0
        return s[BINS]

    # Fallback: derive θ̂ per bin from the curve maximum
    curves = load_bin_curves(site)
    angles = {}
    for b in BINS:
        df = curves[b]
        idx = df["gain_pct"].values.argmax()
        angles[b] = float(df.loc[idx, "deg"])
    return pd.Series(angles)[BINS]

def load_month_shares(site: str) -> pd.DataFrame:
    """
    Expects tables/bin_share_by_month_{SITE}.csv with columns:
    month, month_name, clear, mostly_clear, partly_cloudy, cloudy
    Values may be % (0..100) or fraction (0..1). We convert to fraction.
    """
    f = f"tables/bin_share_by_month_{site}.csv"
    df = pd.read_csv(f)
    df["month"] = df["month"].astype(int)
    for b in BINS:
        if df[b].max() > 1.0 + 1e-6:
            df[b] = df[b] / 100.0
    # For dominance coloring/markers
    df["dominant_bin"] = df[BINS].idxmax(axis=1)
    return df.sort_values("month").reset_index(drop=True)

def load_season_shares(site: str) -> pd.DataFrame:
    """
    Expects tables/bin_share_by_season_{SITE}.csv with columns:
    season, clear, mostly_clear, partly_cloudy, cloudy
    Values may be % or fraction. We convert to fraction and fix order.
    """
    f = f"tables/bin_share_by_season_{site}.csv"
    df = pd.read_csv(f)
    for b in BINS:
        if df[b].max() > 1.0 + 1e-6:
            df[b] = df[b] / 100.0
    order = ["DJF (summer)", "MAM (autumn)", "JJA (winter)", "SON (spring)"]
    df["__order"] = df["season"].map({s: i for i, s in enumerate(order)})
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    df["dominant_bin"] = df[BINS].idxmax(axis=1)
    return df

# ---------- Math on curves ----------

def gain_at_angle(curve_df: pd.DataFrame, angle: float) -> float:
    """
    Linear interpolation of gain (%) at 'angle' over the provided curve (deg, gain_pct).
    """
    x = curve_df["deg"].values
    y = curve_df["gain_pct"].values
    if angle <= x.min():
        return float(y[x.argmin()])
    if angle >= x.max():
        return float(y[x.argmax()])
    return float(np.interp(angle, x, y))

def expected_gain_scheduled(shares: pd.DataFrame, winners: pd.Series, curves: dict) -> pd.Series:
    """
    Scheduled overrides: use each bin's θ̂ (winners) for that bin.
    Expected gain = Σ_b share[b] * gain_at_angle(curve_b, θ̂_b)
    Returns a Series per row of 'shares': scheduled_gain_pct
    """
    out = []
    for _, row in shares.iterrows():
        g = 0.0
        for b in BINS:
            frac = float(row[b])
            theta = float(winners[b])
            g += frac * gain_at_angle(curves[b], theta)
        out.append(g)
    return pd.Series(out, index=shares.index, name="scheduled_gain_pct")

def expected_gain_default_only(shares: pd.DataFrame, winners: pd.Series, curves: dict) -> pd.DataFrame:
    """
    Single default per period: choose default angle = θ̂ of the dominant bin,
    then apply this ONE angle to all bins via their curves, weighted by shares.
    Returns shares copy with ['default_angle_deg','default_only_gain_pct'].
    """
    df = shares.copy()
    angles = []
    gains  = []
    for _, row in df.iterrows():
        dom = row["dominant_bin"]
        angle = float(winners[dom])
        angles.append(angle)
        g = 0.0
        for b in BINS:
            g += float(row[b]) * gain_at_angle(curves[b], angle)
        gains.append(g)
    df["default_angle_deg"]     = angles
    df["default_only_gain_pct"] = gains
    return df

# ---------- Plotters ----------

def plot_month(site: str):
    curves  = load_bin_curves(site)
    winners = load_winners_angles(site)       # θ̂ per bin
    shares  = load_month_shares(site)         # fractions per month

    # Strategy A (scheduled overrides)
    sched = shares.copy()
    sched["scheduled_gain_pct"] = expected_gain_scheduled(shares, winners, curves)
    # Strategy B (single default per month)
    deflt = expected_gain_default_only(shares, winners, curves)

    # Export table
    out_csv = f"tables/scheduled_gain_by_month_{site}.csv"
    both = (sched[["month","month_name","dominant_bin","scheduled_gain_pct"]]
            .merge(deflt[["month","default_angle_deg","default_only_gain_pct"]], on="month", how="left"))
    ensure_parent(out_csv); both.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Plot
    out_fig = f"figs/schedule_vs0_month_scatter_{site}.pdf"
    fig = plt.figure(figsize=(9.6, 5.0))
    ax  = fig.add_subplot(111)

    x    = both["month"].values
    y0   = np.zeros_like(x, dtype=float)
    ax.scatter(x, y0, s=28, color="black", marker="_", label="0° baseline", zorder=2)

    # scheduled (filled) — color/marker by dominant bin
    for b in BINS:
        m = both["dominant_bin"] == b
        if m.any():
            style = BIN_STYLE[b]
            ax.scatter(both.loc[m, "month"], both.loc[m, "scheduled_gain_pct"],
                       s=55, zorder=3, **style)

    # default-only (open markers)
    for b in BINS:
        m = both["dominant_bin"] == b
        if m.any():
            sty = BIN_STYLE[b].copy()
            edge = sty.pop("color")
            mark = sty.pop("marker")
            ax.scatter(both.loc[m, "month"], both.loc[m, "default_only_gain_pct"],
                       s=80, facecolors="none", edgecolors=edge, marker=mark, zorder=3)

    ax.set_xticks(both["month"])
    ax.set_xticklabels(shares["month_name"].str.slice(0,3))
    ax.set_xlabel("Month")
    ax.set_ylabel("Expected gain vs 0° (%)")
    ax.set_title(f"{site}: Monthly expected gain — Bin-scheduled (filled) vs Single default (open) vs 0° (black)\nColor = dominant bin of month")

    ymin = min(-0.1, float(min(both["scheduled_gain_pct"].min(), both["default_only_gain_pct"].min()) - 0.3))
    ymax = float(max(both["scheduled_gain_pct"].max(), both["default_only_gain_pct"].max()) + 0.4)
    ax.set_ylim(ymin, ymax)

    # Legend
    handles = [plt.Line2D([], [], color="black", marker="_", linestyle="None", label="0° baseline")]
    handles += [plt.Line2D([], [], **BIN_STYLE[b], linestyle="None") for b in BINS]
    labels  = ["0° baseline"] + [BIN_STYLE[b]["label"] for b in BINS]
    ax.legend(handles, labels, ncol=3, loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    ensure_parent(out_fig); fig.savefig(out_fig)
    print(f"[OK] wrote {out_fig}")

def plot_season(site: str):
    curves  = load_bin_curves(site)
    winners = load_winners_angles(site)
    shares  = load_season_shares(site)

    # Strategy A
    sched = shares.copy()
    sched["scheduled_gain_pct"] = expected_gain_scheduled(shares, winners, curves)
    # Strategy B
    deflt = expected_gain_default_only(shares, winners, curves)

    out_csv = f"tables/scheduled_gain_by_season_{site}.csv"
    both = (sched[["season","dominant_bin","scheduled_gain_pct"]]
            .merge(deflt[["season","default_angle_deg","default_only_gain_pct"]], on="season", how="left"))
    ensure_parent(out_csv); both.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    order = ["DJF (summer)", "MAM (autumn)", "JJA (winter)", "SON (spring)"]
    xmap  = {s: i+1 for i, s in enumerate(order)}
    both["x"] = both["season"].astype(str).map(xmap)

    out_fig = f"figs/schedule_vs0_season_scatter_{site}.pdf"
    fig = plt.figure(figsize=(8.8, 5.0))
    ax  = fig.add_subplot(111)

    x    = both["x"].values
    y0   = np.zeros_like(x, dtype=float)
    ax.scatter(x, y0, s=35, color="black", marker="_", label="0° baseline", zorder=2)

    for b in BINS:
        m = both["dominant_bin"] == b
        if m.any():
            style = BIN_STYLE[b]
            ax.scatter(both.loc[m, "x"], both.loc[m, "scheduled_gain_pct"],
                       s=75, zorder=3, **style)

    for b in BINS:
        m = both["dominant_bin"] == b
        if m.any():
            sty = BIN_STYLE[b].copy()
            edge = sty.pop("color")
            mark = sty.pop("marker")
            ax.scatter(both.loc[m, "x"], both.loc[m, "default_only_gain_pct"],
                       s=95, facecolors="none", edgecolors=edge, marker=mark, zorder=3)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel("Season (Southern Hemisphere)")
    ax.set_ylabel("Expected gain vs 0° (%)")
    ax.set_title(f"{site}: Seasonal expected gain — Bin-scheduled (filled) vs Single default (open) vs 0° (black)\nColor = dominant bin of season")

    ymin = min(-0.1, float(min(both["scheduled_gain_pct"].min(), both["default_only_gain_pct"].min()) - 0.3))
    ymax = float(max(both["scheduled_gain_pct"].max(), both["default_only_gain_pct"].max()) + 0.4)
    ax.set_ylim(ymin, ymax)

    handles = [plt.Line2D([], [], color="black", marker="_", linestyle="None", label="0° baseline")]
    handles += [plt.Line2D([], [], **BIN_STYLE[b], linestyle="None") for b in BINS]
    labels  = ["0° baseline"] + [BIN_STYLE[b]["label"] for b in BINS]
    ax.legend(handles, labels, ncol=3, loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22)
    ensure_parent(out_fig); fig.savefig(out_fig)
    print(f"[OK] wrote {out_fig}")

def main():
    for site in SITES:
        plot_month(site)
        plot_season(site)

if __name__ == "__main__":
    main()
