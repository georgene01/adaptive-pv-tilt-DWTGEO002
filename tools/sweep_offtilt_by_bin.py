#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, json, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEF_DATE_CANDIDATES = ['date','date_local','day','dt','yyyy_mm_dd','yyyymmdd']
DEF_BIN_CANDIDATES  = ['bin_label','bin_id','bin','label','class','bin_name','bucket','group']

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--site', required=True)
    p.add_argument('--bins', required=True)
    p.add_argument('--daily_dir', required=True)
    p.add_argument('--min_deg', type=int, default=-14)
    p.add_argument('--max_deg', type=int, default=14)
    p.add_argument('--deg_step', type=int, default=2)
    p.add_argument('--y', choices=['energy','gain'], default='energy')
    p.add_argument('--cmd', required=True, help='Use {site},{weather},{date},{min_deg},{max_deg},{deg_step}')
    p.add_argument('--out_bin_table', required=True)
    p.add_argument('--out_bin_fig', required=True)
    p.add_argument('--out_scatter', required=True)
    p.add_argument('--date_col', default=None)
    p.add_argument('--bin_col', default=None)
    return p.parse_args()

def find_col(df, user_col, candidates):
    if user_col and user_col in df.columns: return user_col
    if user_col:
        # case-insensitive fallback
        for c in df.columns:
            if c.lower() == user_col.lower(): return c
        raise SystemExit(f"Requested column '{user_col}' not found. Available: {list(df.columns)}")
    lower = {c.lower():c for c in df.columns}
    for want in candidates:
        if want in lower: return lower[want]
    # fuzzy: anything with 'date' or 'day'
    for c in df.columns:
        if 'date' in c.lower() or 'day' in c.lower(): return c
    return None

def normalize_date(series):
    s = series.copy()
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        s = s.astype('Int64').astype(str)
    s = s.astype(str).str.strip().str.replace(r'[\\/\.]', '-', regex=True)
    mask8 = s.str.fullmatch(r'\d{8}', na=False)
    s.loc[mask8] = s.loc[mask8].str.replace(r'(\d{4})(\d{2})(\d{2})', r'\1-\2-\3', regex=True)
    return pd.to_datetime(s, errors='coerce').dt.strftime('%Y-%m-%d')

def extract_last_json(text: str):
    """Find the last JSON object in stdout by scanning backwards for '{' and attempting json.loads."""
    i = len(text) - 1
    while i >= 0:
        if text[i] == '{':
            chunk = text[i:]
            try:
                return json.loads(chunk)
            except Exception:
                pass
        i -= 1
    return None

def json_to_df(obj):
    # Flexible schema: prefer 'curve', else list, else 'results'
    rows = None
    if isinstance(obj, dict):
        if 'curve' in obj and isinstance(obj['curve'], list):
            rows = obj['curve']
        elif 'results' in obj and isinstance(obj['results'], list):
            rows = obj['results']
        elif 'data' in obj and isinstance(obj['data'], list):
            rows = obj['data']
        else:
            # try values of the dict if they look like rows
            for k, v in obj.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows = v
                    break
    elif isinstance(obj, list):
        rows = obj
    if rows is None:
        raise RuntimeError("Could not interpret JSON from daily runner—no rows found")

    df = pd.DataFrame(rows)
    # Rename common variants
    if 'deg' not in df.columns:
        for cand in ['angle','off_deg','theta','offset_deg']: 
            if cand in df.columns: df = df.rename(columns={cand:'deg'}); break
    if 'E_kWh' not in df.columns:
        for cand in ['energy_kWh','E','energy','Eday_kWh','E_day_kWh']:
            if cand in df.columns: df = df.rename(columns={cand:'E_kWh'}); break
    if 'deg' not in df.columns or 'E_kWh' not in df.columns:
        raise RuntimeError(f"Daily JSON missing required columns. Got columns: {list(df.columns)}")
    return df[['deg','E_kWh']]


def run_one_day(site, date_str, weather_path, min_deg, max_deg, deg_step, cmd_fmt):
    import os, sys, subprocess, pandas as pd
    os.makedirs("tmp/daily_sweep", exist_ok=True)
    cache_csv = f"tmp/daily_sweep/{site}_{date_str}.csv"

    # 0) Cached?
    if os.path.exists(cache_csv):
        try:
            df = pd.read_csv(cache_csv)
            if set(['deg','E_kWh']).issubset(df.columns):
                df['date'] = date_str
                return df[['deg','E_kWh','date']]
        except Exception:
            pass

    # 1) Run the day command
    cmd = cmd_fmt.format(site=site, weather=weather_path, date=date_str,
                         min_deg=min_deg, max_deg=max_deg, deg_step=deg_step)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode != 0:
        print(res.stdout, file=sys.stdout)
        print(res.stderr, file=sys.stderr)
        print(f"[WARN] CMD failed ({res.returncode}): {cmd}", file=sys.stderr)
        return None

    # 2) Prefer CSV the child produced
    if os.path.exists(cache_csv):
        try:
            df = pd.read_csv(cache_csv)
            if set(['deg','E_kWh']).issubset(df.columns):
                df['date'] = date_str
                return df[['deg','E_kWh','date']]
        except Exception as e:
            print(f"[WARN] Could not read produced CSV {cache_csv}: {e}", file=sys.stderr)

    # 3) Last-resort: try to parse JSON from stdout (kept for backwards compat)
    import json
    def extract_last_json(text: str):
        i = len(text) - 1
        while i >= 0:
            if text[i] == '{':
                chunk = text[i:]
                try:
                    return json.loads(chunk)
                except Exception:
                    pass
            i -= 1
        return None

    obj = extract_last_json(res.stdout) or extract_last_json(res.stderr or "")
    if obj is None:
        print(f"[WARN] Could not find JSON in output for {date_str}", file=sys.stderr)
        return None

    # Flexible schema → DataFrame
    import pandas as pd
    rows = None
    if isinstance(obj, dict):
        for key in ('curve','results','data'):
            v = obj.get(key, None)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                rows = v; break
        if rows is None:
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows = v; break
        if rows is None and all(isinstance(k, (int,float)) for k in obj.keys()):
            rows = [{'deg': int(k), 'E_kWh': float(v)} for k,v in sorted(obj.items())]
    elif isinstance(obj, list):
        rows = obj

    if not rows:
        print(f"[WARN] Could not interpret JSON from daily runner—no rows found", file=sys.stderr)
        return None

    df = pd.DataFrame(rows)
    if 'deg' not in df.columns:
        for cand in ['angle','off_deg','theta','offset_deg']:
            if cand in df.columns: df = df.rename(columns={cand:'deg'}); break
    if 'E_kWh' not in df.columns:
        for cand in ['energy_kWh','E','energy','Eday_kWh','E_day_kWh']:
            if cand in df.columns: df = df.rename(columns={cand:'E_kWh'}); break
    if 'deg' not in df.columns or 'E_kWh' not in df.columns:
        print(f"[WARN] Daily results missing required columns for {date_str}", file=sys.stderr)
        return None
    df['date'] = date_str
    # Cache for future runs
    try:
        df[['deg','E_kWh']].to_csv(cache_csv, index=False)
    except Exception:
        pass
    return df[['deg','E_kWh','date']]


def aggregate_bin(site, bin_id, dates, daily_dir, min_deg, max_deg, deg_step, cmd_fmt):
    day_tables = []
    for d in dates:
        weather = os.path.join(daily_dir, f"{site}_{d}.csv")
        if not os.path.exists(weather):
            print(f"[WARN] Missing weather for {d}: {weather}", file=sys.stderr)
            continue
        df_day = run_one_day(site, d, weather, min_deg, max_deg, deg_step, cmd_fmt)
        if df_day is not None:
            day_tables.append(df_day)
    if not day_tables:
        return None, None
    D = pd.concat(day_tables, ignore_index=True)

    # Per-day baseline at 0° for gain
    base = D[D['deg']==0][['date','E_kWh']].rename(columns={'E_kWh':'E0'})
    D = D.merge(base, on='date', how='left')
    D['gain_%_vs0'] = 100.0*(D['E_kWh']-D['E0'])/D['E0']

    S = D.groupby('deg', as_index=False).agg(
        mean_E_kWh=('E_kWh','mean'),
        std_E_kWh=('E_kWh','std'),
        n_obs=('E_kWh','count'),
        mean_gain_pct_vs0=('gain_%_vs0','mean')
    )
    idx = S['mean_E_kWh'].idxmax()
    theta_hat = int(S.loc[idx, 'deg'])
    band = S[(S['deg']>=theta_hat-2)&(S['deg']<=theta_hat+2)]
    flatness = np.nan if band.empty else 100.0*(band['mean_E_kWh'].max()-band['mean_E_kWh'].min())/band['mean_E_kWh'].max()

    S['site'] = site
    S['bin_id'] = bin_id
    S['is_winner'] = (S['deg']==theta_hat).astype(int)
    S['theta_hat'] = theta_hat
    S['flatness_pm2_pct'] = flatness
    S['n_days'] = D['date'].nunique()
    return S, D

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_bin_table), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_bin_fig), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_scatter), exist_ok=True)

    bins_df = pd.read_csv(args.bins)
    bins_df.columns = [c.strip() for c in bins_df.columns]

    date_col = args.date_col or find_col(bins_df, None, DEF_DATE_CANDIDATES)
    bin_col  = args.bin_col  or find_col(bins_df, None, DEF_BIN_CANDIDATES)
    if date_col is None or bin_col is None:
        raise SystemExit(
            "Could not auto-detect date/bin columns.\n"
            f"Available columns: {list(bins_df.columns)}\n"
            f"Try adding --date_col <name> and --bin_col <name>."
        )
    dates_iso = normalize_date(bins_df[date_col])
    if dates_iso.isna().any():
        bad = bins_df.loc[dates_iso.isna(), [date_col]].head(5).to_dict(orient='records')
        raise SystemExit(f"Some dates could not be parsed in '{date_col}'. Examples: {bad}")
    bins_df['__date'] = dates_iso
    bins_df['__bin']  = bins_df[bin_col].astype(str).str.strip()

    all_S = []
    perday_best_rows = []

    for bin_id, g in bins_df.groupby('__bin'):
        dates = list(g['__date'].unique())
        print(f"[INFO] Bin={bin_id}: days={len(dates)}")
        S, D = aggregate_bin(args.site, bin_id, dates, args.daily_dir,
                             args.min_deg, args.max_deg, args.deg_step, args.cmd)
        if S is None:
            print(f"[WARN] No results for bin {bin_id}")
            continue
        all_S.append(S)
        idxmax = D.groupby('date')['E_kWh'].idxmax()
        best = D.loc[idxmax, ['date','deg','E_kWh']].rename(columns={'deg':'best_deg','E_kWh':'best_E_kWh'})
        best['bin_id'] = bin_id
        perday_best_rows.append(best)

    if not all_S:
        raise SystemExit("No results produced for any bin.")

    OUT = pd.concat(all_S, ignore_index=True)
    OUT = OUT[['site','bin_id','n_days','deg','mean_E_kWh','std_E_kWh','mean_gain_pct_vs0','is_winner','theta_hat','flatness_pm2_pct']]
    OUT.sort_values(['site','bin_id','deg'], inplace=True)
    OUT.to_csv(args.out_bin_table, index=False)
    print(f"[OK] wrote {args.out_bin_table}")

    # Bar figure
    bins_order = sorted(OUT['bin_id'].unique())
    n = len(bins_order); cols = 2 if n>1 else 1; rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
    for i, b in enumerate(bins_order):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        S = OUT[OUT['bin_id']==b]
        ax.bar(S['deg'], S['mean_gain_pct_vs0'])
        ax.axvline(int(S['theta_hat'].iloc[0]), linestyle='--', linewidth=1)
        ax.set_title(f"{b} (n_days={int(S['n_days'].iloc[0])})")
        ax.set_xlabel("Off-tilt angle (deg)")
        ax.set_ylabel("Mean gain vs 0° (%)")
        ax.grid(True, alpha=0.3)
    for j in range(i+1, rows*cols):
        r, c = divmod(j, cols); axes[r][c].axis('off')
    fig.tight_layout()
    fig.savefig(args.out_bin_fig, bbox_inches='tight')
    print(f"[OK] wrote {args.out_bin_fig}")

    # Scatter best angle per day
    if perday_best_rows:
        B = pd.concat(perday_best_rows, ignore_index=True)
        B['date'] = pd.to_datetime(B['date'])
        bins_order = sorted(B['bin_id'].unique())
        n = len(bins_order); cols = 1 if n==1 else 2; rows = (n + cols - 1)//cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
        for i, b in enumerate(bins_order):
            r, c = divmod(i, cols)
            ax = axes2[r][c]
            bb = B[B['bin_id']==b].sort_values('date')
            ax.scatter(bb['date'], bb['best_deg'], s=12)
            ax.set_title(f"Best angle per day — {b}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Best off-tilt (deg)")
            ax.grid(True, alpha=0.3)
        for j in range(i+1, rows*cols):
            r, c = divmod(j, cols); axes2[r][c].axis('off')
        fig2.tight_layout()
        fig2.savefig(args.out_scatter, bbox_inches='tight')
        print(f"[OK] wrote {args.out_scatter}")
    else:
        print("[WARN] No per-day best-angle data to plot scatter.")

if __name__ == '__main__':
    main()
