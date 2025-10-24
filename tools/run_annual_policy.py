from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tools.sim_support import build_sim_config, run_daily

ROOT = Path(".")
SEL = ROOT/"selection"
TABLES = ROOT/"tables"
DATA = ROOT/"data_raw"
TABLES.mkdir(exist_ok=True, parents=True)

def load_daylist(site: str) -> pd.DataFrame:
    fn = SEL/f"{site}_2024_day_bins.csv"
    df = pd.read_csv(fn)
    if "date_local" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date":"date_local"})
    if "bin" not in df.columns and "bin_label" in df.columns:
        df = df.rename(columns={"bin_label":"bin"})
    if "season" not in df.columns:
        m = pd.to_datetime(df["date_local"]).dt.month
        season = pd.cut(m, bins=[0,2,5,8,11,12],
                        labels=["Summer (DJF)","Autumn (MAM)","Winter (JJA)","Spring (SON)","Summer (DJF)"],
                        right=True, include_lowest=True)
        df["season"] = season.values
    df["site"]=site
    return df[["site","date_local","bin","season"]].copy()

def site_weather(site: str) -> str:
    return str(DATA/f"{site}_2024_POWER_qc.csv")

def bin_median_offsets() -> pd.DataFrame:
    summ = pd.read_csv(TABLES/"daily_energy_all.csv")
    summ["bin_code"] = summ["bin"].str.split(":").str[0]
    g = summ.groupby(["site","bin_code"])
    med = g[["offset_opt_deg","delta_pct"]].median().reset_index()
    med = med.rename(columns={"bin_code":"bin","offset_opt_deg":"theta_med_deg","delta_pct":"delta_med_pct"})
    return med

def compute_day_energy(site: str, date: str, delta_deg: float) -> float:
    sim = build_sim_config(site, base_dir=".")
    out = run_daily(sim, site_weather(site), date_local=date, offset_deg=float(delta_deg))
    return float(out["pac"].clip(lower=0).sum()/1000.0)  # kWh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scada", type=str, default=None, help="Optional SCADA daily CSV with columns: site,date_local,energy_kWh")
    args = ap.parse_args()

    # If you only want validation (no SA sites set up), we can skip NC/WC gracefully
    have_nc = (SEL/"NC_2024_day_bins.csv").exists()
    have_wc = (SEL/"WC_2024_day_bins.csv").exists()
    daylists = []
    if have_nc: daylists.append(load_daylist("NC"))
    if have_wc: daylists.append(load_daylist("WC"))
    if daylists:
        daylists = pd.concat(daylists, ignore_index=True)
        daylists["date_local"] = pd.to_datetime(daylists["date_local"]).dt.strftime("%Y-%m-%d")

    # Bin medians for SA sites (if present)
    med = bin_median_offsets() if (TABLES/"daily_energy_all.csv").exists() else pd.DataFrame(columns=["site","bin","theta_med_deg","delta_med_pct"])

    pol_rows, daily_rows = [], []

    # Run SA (NC/WC) annual policy comparison if we have day lists
    for site, g in (daylists.groupby("site") if isinstance(daylists, pd.DataFrame) and not daylists.empty else []):
        th_map = med[med["site"]==site].set_index("bin")["theta_med_deg"].to_dict()
        b4_theta = th_map.get("B4", 0.0)
        E0 = E_b4 = E_bin = 0.0
        for _, r in g.iterrows():
            date = r["date_local"]
            b = (r["bin"].split(":")[0] if ":" in r["bin"] else r["bin"]).strip()
            e0 = compute_day_energy(site, date, 0.0)
            eb4 = compute_day_energy(site, date, b4_theta if b=="B4" else 0.0)
            ebin = compute_day_energy(site, date, th_map.get(b, 0.0))
            E0 += e0; E_b4 += eb4; E_bin += ebin
            daily_rows.append({"site":site,"date":date,"bin":b,
                               "E0_kWh":e0,"E_policyB4_kWh":eb4,"E_idealBin_kWh":ebin,
                               "theta_B4_deg":b4_theta,"theta_bin_deg":th_map.get(b,0.0)})
        if E0>0:
            pol_rows.append({"site":site,
                             "E0_year_kWh":E0,
                             "E_policyB4_year_kWh":E_b4,
                             "E_idealBin_year_kWh":E_bin,
                             "gain_policyB4_pct":100*(E_b4-E0)/E0,
                             "gain_idealBin_pct":100*(E_bin-E0)/E0})

    if daily_rows:
        daily = pd.DataFrame(daily_rows)
        daily.to_csv(TABLES/"annual_policy_daily_log.csv", index=False)
    if pol_rows:
        annual = pd.DataFrame(pol_rows)
        annual.to_csv(TABLES/"annual_policy_by_site.csv", index=False)
        print("\nAnnual energy by site (kWh) and gains vs baseline (%):")
        print(annual.to_string(index=False))
    else:
        print("\nNo NC/WC annual policy run (day lists not found) — proceeding to SCADA validation only.")

    # SCADA validation (baseline model vs measured)
    if args.scada:
        sc = pd.read_csv(args.scada)
        sc["date_local"]=pd.to_datetime(sc["date_local"]).dt.strftime("%Y-%m-%d")
        # Validate for any site present in SCADA that also has a row in inputs/sites.csv
        sites = sc["site"].unique().tolist()
        vals = []
        for site in sites:
            # we need the site to exist in inputs/sites.csv so build_sim_config can run
            try:
                # model daily AC for all scada dates
                dates = sc[sc["site"]==site]["date_local"].tolist()
                E0s = []
                for d in dates:
                    try:
                        E0s.append({"date_local":d, "E0_kWh": compute_day_energy(site, d, 0.0)})
                    except Exception:
                        # if any missing weather day, skip that date
                        pass
                if not E0s: 
                    continue
                mod = pd.DataFrame(E0s)
                m = mod.merge(sc[sc["site"]==site][["date_local","energy_kWh"]], on="date_local", how="inner")
                if m.empty:
                    continue
                y = m["energy_kWh"].to_numpy()
                yhat = m["E0_kWh"].to_numpy()
                mbe = float((yhat - y).mean())
                nrmse = float(np.sqrt(np.mean((yhat-y)**2)) / (y.mean() if y.mean()!=0 else 1.0))
                vals.append({"site":site,"n_days":len(m),"MBE_kWh":mbe,"NRMSE_frac":nrmse})
            except Exception as e:
                continue
        if vals:
            v = pd.DataFrame(vals)
            v.to_csv(TABLES/"validity_stats.csv", index=False)
            print("\nValidation vs SCADA (baseline model → measured):")
            print(v.to_string(index=False))
        else:
            print("\nSCADA validation ran, but no overlapping dates/site config were usable (check inputs/sites.csv and weather files).")
    else:
        print("\nSCADA file not provided: run with --scada scada_daily_template.csv when ready.")
if __name__ == "__main__":
    main()
