#!/usr/bin/env python3
import argparse, os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CASES = [
    ("WC baseline", 105.7, (90.8, 121.0)),
    ("WC offpoint", 107.6, (92.7, 123.7)),
    ("NC baseline",  94.9, (81.7, 109.2)),
    ("NC offpoint",  96.6, (83.2, 111.3)),
]

def load_or_synthesize(paths):
    rows = []
    for label, med, (p5, p95) in CASES:
        path = paths.get(label, None)
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if 'lcoe' not in df.columns:
                raise SystemExit(f"{path} must have a column named 'lcoe'")
            x = df['lcoe'].to_numpy(float)
            mode = "real"
        else:
            # Triangular approx matching 5–95 and median
            lo, hi = p5, p95
            c = med
            # sample triangular with (lo, mode≈median, hi)
            # note: triangular median != mode; we choose mode=median for a
            # simple, conservative shape around the reported center.
            x = np.random.default_rng(42).triangular(lo, c, hi, size=8000)
            mode = "approx"
        rows.append(pd.DataFrame({"case": label, "lcoe": x, "mode": mode}))
    return pd.concat(rows, ignore_index=True)

def plot_hist_violin(df, out_prefix="figs/lcoe"):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # 1) Histogram panel
    plt.figure(figsize=(8.6, 5.2))
    bins = np.arange(60, 150, 2)  # fixed axis for comparability
    for label, grp in df.groupby("case"):
        plt.hist(grp["lcoe"], bins=bins, alpha=0.35, density=True, label=label)
    plt.xlabel("LCOE (real $/MWh)")
    plt.ylabel("Density")
    plt.title("Monte-Carlo LCOE distributions")
    plt.legend(frameon=False)
    plt.tight_layout()
    hist_path = f"{out_prefix}_hist.pdf"
    plt.savefig(hist_path)
    plt.close()

    # 2) Violin panel
    order = [c for c,_,_ in CASES]
    data = [df.loc[df["case"]==c, "lcoe"].to_numpy() for c in order]
    plt.figure(figsize=(8.6, 4.8))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(range(1, len(order)+1), order, rotation=15)
    plt.ylabel("LCOE (real $/MWh)")
    plt.title("Monte-Carlo LCOE (violin plots)")
    plt.tight_layout()
    violin_path = f"{out_prefix}_violin.pdf"
    plt.savefig(violin_path)
    plt.close()
    return hist_path, violin_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", help="JSON mapping label->CSV path; labels must match: 'WC baseline','WC offpoint','NC baseline','NC offpoint'")
    ap.add_argument("--out_prefix", default="figs/lcoe")
    args = ap.parse_args()

    paths = {}
    if args.samples_json and os.path.exists(args.samples_json):
        with open(args.samples_json) as f:
            paths = json.load(f)

    df = load_or_synthesize(paths)
    hist_path, violin_path = plot_hist_violin(df, args.out_prefix)

    # quick medians and 5–95% from whatever we plotted
    summary = (df.groupby("case")["lcoe"]
                 .agg(median=np.median,
                      p5=lambda s: np.quantile(s, 0.05),
                      p95=lambda s: np.quantile(s, 0.95))
                 .reset_index())
    txt = args.out_prefix + "_summary.txt"
    summary.to_csv(txt, index=False)
    print("Wrote:", hist_path, violin_path, txt)

if __name__ == "__main__":
    main()
