#!/usr/bin/env python3
"""
Compute multi-metric sensitivity of LCOE to input drivers.

Input CSV must contain at least:
  - lcoe               (real $/MWh)
Plus as many of these as you have (names are flexible; we fuzzy-match):
  - capex per kWac           -> capex, capex_$perkWac, capex_kwac, capex_ac
  - real WACC                -> wacc, wacc_real
  - fixed O&M per kWac-year  -> fixed_om, fixed_om_$perkWac_yr, fom
  - availability (%)         -> availability, availability_pct, avail
  - first-year degradation (%) or annual mean -> degradation, degradation_%
  - annual energy (MWh)      -> annual_mwh, ac_mwh, energy_mwh, yield_mwh
  - dc/ac ratio              -> dcac, dcac_ratio
  - inverter efficiency      -> inverter_eff, eta_inv
  - dc system efficiency     -> dc_system_eff, eta_dc
  - clipping energy (MWh)    -> clip_mwh
  - anything else you sampled (ok!)

Outputs:
  - CSV with all metrics, ranked by “consensus_score”
  - Tornado PDF/PNG
"""

import argparse, os, re, math, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ---------- helpers ----------
CANDIDATES = {
    "capex_$perkWac":   [r"capex", r"capex.*kwac", r"capex_.*ac", r"capex_\$perkWac"],
    "wacc_real":        [r"wacc.*real", r"^wacc$"],
    "fixed_om_$perkWac_yr":[r"fixed[_ ]?o.?m", r"fom", r"fixed_om.*kwac"],
    "availability_%":   [r"availability", r"avail"],
    "degradation_%":    [r"degrad"],
    "annual_MWh":       [r"annual.*mwh", r"\bac_mwh\b", r"energy_mwh", r"yield_mwh"],
    "dcac_ratio":       [r"dcac"],
    "inverter_eff":     [r"inverter.*eff", r"eta[_]?inv"],
    "dc_system_eff":    [r"dc[_ ]?system.*eff", r"eta[_]?dc"],
    "clip_MWh":         [r"clip.*mwh"],
}

def pick_columns(df):
    cols = {"lcoe":"lcoe"}
    if "lcoe" not in df.columns:
        raise SystemExit("CSV must have a 'lcoe' column (real $/MWh).")
    for canonical, pats in CANDIDATES.items():
        for c in df.columns:
            lc = c.lower()
            if any(re.search(p, lc) for p in pats):
                cols[canonical] = c
                break
    # also collect any “unknown” numeric drivers you sampled
    known = set(cols.values())
    extra = [c for c in df.columns if c not in known and c!="lcoe" and pd.api.types.is_numeric_dtype(df[c])]
    return cols, extra

def _safe_log(x):
    # avoid log of nonpositive (drop or shift tiny)
    x = np.asarray(x, float)
    m = np.nanmin(x)
    if m <= 0:
        x = x - m + 1e-6
    return np.log(x)

def prcc(dfX, y):
    """Partial Rank Correlation Coefficients via residual method."""
    # rank-transform
    Xr = dfX.apply(rankdata, axis=0, raw=True)
    yr = rankdata(y, method="average")
    pr = {}
    for j, col in enumerate(dfX.columns):
        others = [c for c in dfX.columns if c != col]
        if not others:
            pr[col] = np.nan
            continue
        # regress col~others, y~others (linear on ranks)
        lr1 = LinearRegression().fit(Xr[others], Xr[col])
        lr2 = LinearRegression().fit(Xr[others], yr)
        r1 = Xr[col] - lr1.predict(Xr[others])
        r2 = yr      - lr2.predict(Xr[others])
        pr[col] = np.corrcoef(r1, r2)[0,1]
    return pr

def standardize_and_fit(X, y):
    scX = StandardScaler()
    scy = StandardScaler()
    Xs = scX.fit_transform(X)
    ys = scy.fit_transform(y.reshape(-1,1)).ravel()
    lr = LinearRegression().fit(Xs, ys)
    return dict(zip(X.columns, lr.coef_))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Monte-Carlo samples CSV (inputs + lcoe)")
    ap.add_argument("--label", default="LCOE case", help="Label for plots")
    ap.add_argument("--out_prefix", default="figs/sensitivity")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv)
    colmap, extra = pick_columns(df)

    y = df[colmap["lcoe"]].to_numpy(float) if "lcoe" in colmap else df["lcoe"].to_numpy(float)

    # Build driver matrix from all known + extras
    drivers = [v for k,v in colmap.items() if k!="lcoe"]
    drivers += extra
    drivers = list(dict.fromkeys(drivers))  # unique preserve order
    X = df[drivers].copy()

    # Drop rows with any NaN among selected columns
    keep = ~pd.isna(X).any(axis=1) & ~pd.isna(y)
    X = X.loc[keep]
    y = y[keep.values]

    # Pearson/Spearman
    pearson = X.corrwith(pd.Series(y, index=X.index), numeric_only=True).to_dict()
    spearman_vals = {}
    for c in X.columns:
        v = spearmanr(X[c].to_numpy(float), y)[0]
        spearman_vals[c] = v

    # Standardized linear betas
    betas = standardize_and_fit(X, y)

    # Elasticities (log–log). Drop nonpositive columns
    elastic = {}
    poscols = [c for c in X.columns if (X[c] > 0).all() and (y > 0).all()]
    if len(poscols) >= 1:
        Xlog = X[poscols].apply(_safe_log)
        ylog = _safe_log(y)
        el = standardize_and_fit(Xlog, ylog)  # standardized slope ~ elasticity when log–log
        elastic.update(el)

    # PRCC
    pr = prcc(X, y)

    # Permutation importance via RF (nonparametric, captures interactions)
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1
    ).fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    perm_imp = dict(zip(X.columns, perm.importances_mean))

    # Assemble table
    rows = []
    for c in X.columns:
        rows.append({
            "driver": c,
            "pearson_r": pearson.get(c, np.nan),
            "spearman_r": spearman_vals.get(c, np.nan),
            "beta_std_lin": betas.get(c, np.nan),
            "elasticity_loglog": elastic.get(c, np.nan),
            "prcc": pr.get(c, np.nan),
            "perm_importance": perm_imp.get(c, np.nan),
        })
    res = pd.DataFrame(rows)

    # Rank by “consensus”: mean of absolute z-scores across metrics (robust)
    def z(x):
        m, s = np.nanmean(x), np.nanstd(x)
        return (x - m)/s if s>0 else np.zeros_like(x)
    metrics = ["pearson_r","spearman_r","beta_std_lin","elasticity_loglog","prcc","perm_importance"]
    Z = np.column_stack([np.abs(z(res[m].to_numpy(float))) for m in metrics])
    res["consensus_score"] = np.nanmean(Z, axis=1)

    res = res.sort_values("consensus_score", ascending=False)
    out_csv = f"{args.out_prefix}_table.csv"
    res.to_csv(out_csv, index=False)

    # Tornado: use signed PRCC if available; fallback to signed Pearson
    metric = "prcc" if res["prcc"].notna().any() else "pearson_r"
    top = res.head(args.topk).copy()
    top = top.sort_values(metric)

    plt.figure(figsize=(7.2, 4.6))
    ylab = top["driver"].tolist()
    vals = top[metric].to_numpy(float)
    colors = ["tab:red" if v>0 else "tab:blue" for v in vals]
    y_pos = np.arange(len(vals))
    plt.barh(y_pos, vals, edgecolor="black", linewidth=0.5, color=colors, alpha=0.85)
    plt.yticks(y_pos, ylab)
    plt.xlabel(f"Sensitivity to LCOE ({metric})")
    plt.title(f"{args.label}: Tornado (top {len(vals)})")
    plt.axvline(0, color="k", linewidth=0.8)
    plt.tight_layout()
    pdf = f"{args.out_prefix}_tornado.pdf"
    png = f"{args.out_prefix}_tornado.png"
    plt.savefig(pdf)
    plt.savefig(png, dpi=200)
    plt.close()

    print(f"Wrote:\n  {out_csv}\n  {pdf}\n  {png}")
    # Quick console summary
    print("\nTop (consensus):")
    print(res.head(args.topk)[["driver","consensus_score",metric,"perm_importance"]].to_string(index=False))

if __name__ == "__main__":
    main()
