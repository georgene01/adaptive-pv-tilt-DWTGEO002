#!/usr/bin/env python3
import numpy as np
import argparse, os, numpy as np, pandas as pd

def read_y0_mwh(path):
    df = pd.read_csv(path)
    ac_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("ac_kwh_day","e_ac_clip_kwh","ac_kwh"):
            ac_col = c; break
    if ac_col is None:
        raise SystemExit(f"Could not find an AC energy column in {path}")
    return float(df[ac_col].sum()/1000.0)

def crf(r, n):
    if r == 0: return 1.0/n
    return r*(1+r)**n / ((1+r)**n - 1)

def pv_annuity(c, r, n, grow=0.0, first_at_t1=True):
    r = float(r); g = float(grow)
    pv = 0.0
    for t in range(1, n+1):
        ct = c*((1+g)**(t-1))
        pv += ct/((1+r)**t if first_at_t1 else (1+r)**(t-1))
    return pv

def pv_energy(y0_mwh, r, n, degr):
    pv = 0.0
    for t in range(1, n+1):
        Et = y0_mwh*((1.0-degr)**(t-1))
        pv += Et/((1+r)**t)
    return pv

def mc_lcoe(y0_mwh, N=5000, seed=42):
    rng = np.random.default_rng(seed)
    lifetime_y = rng.integers(25, 31, size=N)
    wacc_real  = rng.normal(0.055, 0.01, size=N).clip(0.03,0.09)
    capex_kwac = rng.normal(950.0, 120.0, size=N).clip(700,1300)
    fom_kwac_y = rng.normal(17.0, 5.0, size=N).clip(8,40)
    availability = rng.normal(0.985, 0.008, size=N).clip(0.95, 0.995)
    degr = rng.normal(0.005, 0.002, size=N).clip(0.001,0.01)
    pac_kw = 50000.0

    y0_net = y0_mwh * availability
    capex_total = capex_kwac * pac_kw
    pv_fom = np.array([pv_annuity(fom_kwac_y[i]*pac_kw, wacc_real[i], int(lifetime_y[i])) for i in range(N)])
    pv_E   = np.array([pv_energy(y0_net[i], wacc_real[i], int(lifetime_y[i]), degr[i]) for i in range(N)])
    lcoe = capex_total/pv_E + pv_fom/pv_E

    import pandas as pd
    return pd.DataFrame({
        "lcoe": lcoe,
        "y0_mwh": y0_mwh,
        "y0_net_mwh": y0_net,
        "availability_%": availability*100.0,
        "degradation_%": degr*100.0,
        "wacc_real": wacc_real,
        "capex_$perkWac": capex_kwac,
        "fixed_om_$perkWac_yr": fom_kwac_y,
        "lifetime_y": lifetime_y,
        "PAC_kWac": pac_kw,
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    y0 = read_y0_mwh(args.energy_csv)
    df = mc_lcoe(y0, N=args.n, seed=args.seed)
    df["case_label"] = args.label
    df.to_csv(args.out_csv, index=False)
    p5,p50,p95 = np.percentile(df["lcoe"], [5,50,95])
    print(f"{args.label or os.path.basename(args.out_csv)}  p50={p50:5.1f}  (p5={p5:5.1f}, p95={p95:5.1f})  $/MWh (real)")

if __name__ == "__main__":
    main()
