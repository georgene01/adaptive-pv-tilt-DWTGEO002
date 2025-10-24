#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sim_support import build_sim_config, run_daily

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scada", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--offset-deg", type=float, default=0.0)
    ap.add_argument("--scale-sim", type=float, default=1.0)
    ap.add_argument("--auto-scale", action="store_true")
    return ap.parse_args()

def load_scada(path):
    df = pd.read_csv(path)
    for req in ["site","date_local","energy_kWh"]:
        if req not in df.columns: raise RuntimeError(f"SCADA missing {req}")
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["E_scada_Wh"] = pd.to_numeric(df["energy_kWh"], errors="coerce")*1000.0
    return df[["site","date_local","E_scada_Wh"]]

def main():
    a = parse_args()
    os.makedirs("tables", exist_ok=True); os.makedirs("figs", exist_ok=True)

    # Build sim for PVDAQ1430 (Denver) using your single-axis tracker
    simcfg = build_sim_config(a.site, base_dir=".")
    scada = load_scada(a.scada)

    # Simulate each day in 2017 (dates from SCADA)
    rows=[]
    for _, r in scada.iterrows():
        d = r["date_local"].isoformat()
        try:
            out = run_daily(simcfg, a.weather, d, offset_deg=a.offset_deg)  # <-- your tracker
            e_sim = float(out["e_day_wh"])
        except Exception:
            e_sim = np.nan
        rows.append({"date_local":d,"E_scada_Wh":float(r["E_scada_Wh"]), "E_sim_Wh_raw":e_sim})
    res = pd.DataFrame(rows)

    # Auto-scale if requested
    scale = float(a.scale_sim)
    if a.auto_scale:
        calib = res.dropna(subset=["E_scada_Wh","E_sim_Wh_raw"]).copy()
        if not calib.empty and (calib["E_sim_Wh_raw"]>0).any():
            s = (calib.E_sim_Wh_raw*calib.E_scada_Wh).sum() / (calib.E_sim_Wh_raw**2).sum()
            if np.isfinite(s) and s>0: scale = float(s)

    res["E_sim_Wh"] = res["E_sim_Wh_raw"] * scale
    res["diff_Wh"] = res["E_sim_Wh"] - res["E_scada_Wh"]
    res["date_local"] = pd.to_datetime(res["date_local"]).dt.date
    res.to_csv(f"tables/validation_{a.site}.csv", index=False)

    # Metrics
    val = res.dropna(subset=["E_scada_Wh","E_sim_Wh"]).copy()
    metrics={}
    if not val.empty:
        y, yhat = val["E_scada_Wh"].values, val["E_sim_Wh"].values
        err = yhat - y
        mbe = err.mean(); rmse = np.sqrt((err**2).mean())
        r2 = 1 - ((err**2).sum())/((y-y.mean())**2).sum() if len(y)>1 else np.nan
        nrmse = rmse / (y.mean() if y.mean()!=0 else 1.0)
        metrics = {"n":int(len(val)), "scale_applied":float(scale),
                   "MBE_Wh":float(mbe), "NMBE_pct":float(100*mbe/(y.mean() or 1.0)),
                   "RMSE_Wh":float(rmse), "NRMSE_pct":float(100*nrmse), "R2":float(r2)}
    pd.DataFrame([metrics]).to_csv(f"tables/validation_metrics_{a.site}.csv", index=False)

    # Plots
    if not val.empty:
        x = val["E_scada_Wh"]/1000.0; yk = val["E_sim_Wh"]/1000.0
        lim=[0, max(x.max(), yk.max())*1.05]
        fig,ax=plt.subplots(figsize=(5,5)); ax.scatter(x,yk,s=45); ax.plot(lim,lim,"--",lw=1,color="gray")
        if len(x)>=2:
            m,b=np.polyfit(x,yk,1); xx=np.linspace(*lim,200); ax.plot(xx,m*xx+b,lw=1)
        if metrics:
            ax.text(0.02,0.98,f"scale={scale:.2f}\n"
                              f"n={metrics['n']}\n"
                              f"R²={metrics['R2']:.3f}\n"
                              f"NMBE={metrics['NMBE_pct']:.1f}%\n"
                              f"NRMSE={metrics['NRMSE_pct']:.1f}%",
                    transform=ax.transAxes, va="top")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("SCADA $E_{day}$ [kWh]"); ax.set_ylabel("Simulated $E_{day}$ [kWh]")
        ax.set_title(f"Daily Energy Validation — {a.site}")
        fig.tight_layout(); fig.savefig(f"figs/validation_scatter_{a.site}.pdf", bbox_inches="tight"); plt.close(fig)

        vts = res.sort_values("date_local").copy()
        vts["date_local"] = pd.to_datetime(vts["date_local"])
        fig,ax=plt.subplots(figsize=(8.5,3.6))
        ax.plot(vts["date_local"], vts["E_scada_Wh"]/1000.0, marker="o", label="SCADA")
        ax.plot(vts["date_local"], vts["E_sim_Wh"]/1000.0, marker="s", label=f"Sim × {scale:.2f}")
        ax.set_ylabel("$E_{day}$ [kWh]"); ax.set_xlabel("Date (local)"); ax.legend()
        ax.set_title(f"{a.site} — Daily Energy (SCADA vs Sim)")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"figs/validation_timeseries_{a.site}.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"Wrote: tables/validation_{a.site}.csv")
    if metrics: print("Metrics:", metrics)
    print(f"Scatter: figs/validation_scatter_{a.site}.pdf")
    print(f"Timeseries: figs/validation_timeseries_{a.site}.pdf")

if __name__ == "__main__": main()
