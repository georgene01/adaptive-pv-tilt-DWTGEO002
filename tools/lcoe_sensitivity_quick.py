import argparse, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

def load_and_clean(csv):
    df = pd.read_csv(csv)
    if "lcoe" not in df.columns:
        raise ValueError("Expected a column named 'lcoe' in the samples CSV.")
    num = df.select_dtypes(include=[np.number]).copy()
    y = num["lcoe"].to_numpy()
    X = num.drop(columns=["lcoe"])
    keep = []
    for c in X.columns:
        v = np.nanvar(X[c].to_numpy())
        if np.isfinite(v) and v > 1e-12:
            keep.append(c)
    X = X[keep].copy()
    X = X.loc[:, ~X.columns.duplicated()].copy()
    return X, y

def spearman_fast(X, y):
    ry = pd.Series(y).rank(method="average").to_numpy()
    rys = (ry - ry.mean()) / ry.std(ddof=0)
    out = []
    for c in X.columns:
        rx = X[c].rank(method="average").to_numpy()
        rxs = (rx - rx.mean()) / rx.std(ddof=0)
        if np.isfinite(rxs).all() and np.isfinite(rys).all() and rxs.std() > 0:
            rho = float(np.corrcoef(rxs, rys)[0,1])
        else:
            rho = np.nan
        out.append((c, rho))
    s = pd.DataFrame(out, columns=["feature","spearman_rho"]).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    return s

def standardized_betas(X, y):
    Xz = (X - X.mean())/X.std(ddof=0)
    yz = (y - y.mean())/y.std(ddof=0)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yz = np.nan_to_num(yz, nan=0.0, posinf=0.0, neginf=0.0)
    lr = LinearRegression()
    lr.fit(Xz, yz)
    betas = pd.Series(lr.coef_, index=Xz.columns, name="beta_std")
    return betas.reindex(betas.abs().sort_values(ascending=False).index)

def light_permutation(X, y, random_state=42):
    rf = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=8, random_state=random_state, n_jobs=1)
    imp = pd.DataFrame({
        "feature": X.columns,
        "perm_importance": perm.importances_mean,
        "perm_importance_std": perm.importances_std
    }).sort_values("perm_importance", ascending=False)
    return imp

def save_bar(df, xcol, ycol, out_png, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, max(3, 0.28*len(df))))
    plt.barh(df[xcol], df[ycol])
    plt.gca().invert_yaxis()
    plt.xlabel(ycol); plt.title(title); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="samples CSV produced by mc_lcoe_from_energy.py")
    ap.add_argument("--label", default="", help="label for titles")
    ap.add_argument("--out_prefix", required=True, help="fig/CSV prefix")
    ap.add_argument("--with_perm", action="store_true", help="also run a light permutation importance")
    args = ap.parse_args()

    X, y = load_and_clean(args.csv)

    s = spearman_fast(X, y)
    b = standardized_betas(X, y).reset_index()
    b.columns = ["feature","beta_std"]

    s.to_csv(f"{args.out_prefix}_spearman.csv", index=False)
    b.to_csv(f"{args.out_prefix}_betas.csv", index=False)
    save_bar(s.head(20), "feature", "spearman_rho", f"{args.out_prefix}_spearman.png", f"{args.label} — Spearman (top 20)")
    save_bar(b.head(20), "feature", "beta_std",     f"{args.out_prefix}_betas.png",    f"{args.label} — Std. betas (top 20)")

    if args.with_perm:
        imp = light_permutation(X, y)
        imp.to_csv(f"{args.out_prefix}_perm.csv", index=False)
        save_bar(imp.head(20), "feature", "perm_importance", f"{args.out_prefix}_perm.png", f"{args.label} — Permutation (top 20)")

    print(f"Done: {args.out_prefix}_spearman.csv/.png, {args.out_prefix}_betas.csv/.png" + (" + permutation" if args.with_perm else ""))

if __name__ == "__main__":
    main()
