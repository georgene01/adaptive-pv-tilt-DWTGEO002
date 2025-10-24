#!/usr/bin/env python3
import argparse, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pytz

def season_from_month(m: int) -> str:
    if m in (12,1,2): return "Summer (DJF)"
    if m in (3,4,5):  return "Autumn (MAM)"
    if m in (6,7,8):  return "Winter (JJA)"
    return "Spring (SON)"

def classify_day(r_med, ws_med, ta_med, wind_split, hot_split):
    if pd.isna(r_med):   sky = "mixed"
    elif r_med >= 0.8:   sky = "clear"
    elif r_med >= 0.4:   sky = "mixed"
    else:                sky = "overcast"
    wind  = "low"  if (pd.notna(ws_med) and ws_med <  wind_split) else "high"
    therm = "hot"  if (pd.notna(ta_med)  and ta_med  >= hot_split) else "cool"
    return sky, wind, therm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="*_LOCAL.csv (has time_local & local_date)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--wind_split_ms", type=float, default=3.0)
    ap.add_argument("--hot_split_c",   type=float, default=25.0)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise FileNotFoundError(args.inp)

    df = pd.read_csv(args.inp)
    # Columns guard
    for c in ["time_utc","time_local","local_date"]: 
        if c not in df.columns: raise RuntimeError(f"'{c}' missing in {args.inp} — use *_LOCAL.csv from cleaner")
    for c in ["ghi_wm2","temp_air_c","wind_speed_ms"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Adaptive daylight threshold (based on scaled data)
    if df["ghi_wm2"].notna().any():
        q90 = np.nanpercentile(df["ghi_wm2"], 90)
        daylight_thr = max(20.0, 0.10*q90)
    else:
        raise RuntimeError("No valid GHI values")
    df["daylight"] = df["ghi_wm2"] > daylight_thr

    # Per-day q95 proxy using daylight only
    day = df[df["daylight"]].copy()
    if day.empty: raise RuntimeError("No daylight rows found")
    q95 = day.groupby("local_date")["ghi_wm2"].quantile(0.95).rename("q95")
    df = df.merge(q95, left_on="local_date", right_index=True, how="left")
    df["q95"] = np.clip(df["q95"], 1.0, None)
    df["r"] = np.where(df["daylight"], df["ghi_wm2"]/df["q95"], np.nan)

    # Daily aggregates (daylight-only)
    agg = df[df["daylight"]].groupby("local_date").agg(
        r_median=("r","median"),
        wind_median=("wind_speed_ms","median"),
        Ta_median=("temp_air_c","median"),
        ghi_day_Whm2=("ghi_wm2","sum")
    ).reset_index()
    agg["ghi_day_MJm2"] = agg["ghi_day_Whm2"]*0.0036
    agg["month"]  = pd.to_datetime(agg["local_date"]).dt.month
    agg["season"] = agg["month"].apply(season_from_month)

    # Labels to bins
    def bin_label(s,w,t):
        if s=="clear"   and w=="high" and t=="hot":  return "B2: Clear / High-wind / Hot"
        if s=="mixed"   and w=="low"  and t=="hot":  return "B3: Mixed / Low-wind / Hot"
        if s=="mixed"   and w=="high" and t=="cool": return "B4: Mixed / High-wind / Cool"
        if s=="overcast"and w=="low"  and t=="cool": return "B5: Overcast / Low-wind / Cool"
        # allow other combos but mark as None (excluded)
        return None

    sky, wind, therm = [], [], []
    for _, row in agg.iterrows():
        s, w, t = classify_day(row["r_median"], row["wind_median"], row["Ta_median"],
                               args.wind_split_ms, args.hot_split_c)
        sky.append(s); wind.append(w); therm.append(t)
    agg["sky"]=sky; agg["wind"]=wind; agg["thermal"]=therm
    agg["bin"] = [bin_label(s,w,t) for s,w,t in zip(agg["sky"], agg["wind"], agg["thermal"])]

    # Save all days with a bin
    all_days = agg[["local_date","bin","season","r_median","wind_median","Ta_median","ghi_day_MJm2"]].copy()
    all_days = all_days.dropna(subset=["bin"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_days.to_csv(args.out.replace(".csv","_ALL.csv"), index=False)
    print(f"[{args.site}] wrote: {args.out.replace('.csv','_ALL.csv')}  (rows={len(all_days)})")

    # Exemplar per (bin, season): highest GHI day (not used by off-tilt but kept)
    cand = all_days.copy()
    cand["rank"] = cand.groupby(["bin","season"])["ghi_day_MJm2"].rank(ascending=False, method="first")
    selected = cand[cand["rank"]==1].copy().sort_values(["bin","season"])
    selected.to_csv(args.out, index=False, columns=["local_date","bin","season","r_median","wind_median","Ta_median","ghi_day_MJm2"])

    # Compact table figure
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.3*max(1,len(selected))))
    ax.axis('off')
    header = ["Date","Bin","Season","Median r","Median WS [m/s]","Median Ta [°C]","GHI day [MJ/m²]"]
    data = [header] + [[str(r["local_date"]), r["bin"], r["season"],
                        f'{r["r_median"]:.2f}' if pd.notna(r["r_median"]) else "",
                        f'{r["wind_median"]:.2f}' if pd.notna(r["wind_median"]) else "",
                        f'{r["Ta_median"]:.1f}' if pd.notna(r["Ta_median"]) else "",
                        f'{r["ghi_day_MJm2"]:.1f}' if pd.notna(r["ghi_day_MJm2"]) else ""] 
                       for _, r in selected.iterrows()]
    tbl = ax.table(cellText=data, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.4)
    fig.tight_layout(); fig.savefig(args.fig, bbox_inches="tight"); plt.close(fig)
    print(f"[{args.site}] wrote: {args.out}")
    print(f"[{args.site}] wrote: {args.fig}")

if __name__=="__main__": main()
