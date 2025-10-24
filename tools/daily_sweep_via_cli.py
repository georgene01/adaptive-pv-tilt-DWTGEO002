#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, json, subprocess, re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runner', default='tools/run_step2_daily.py',
                   help='Path to run_step2_daily.py')
    p.add_argument('--site', required=True)
    p.add_argument('--weather', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--min_deg', type=int, required=True)
    p.add_argument('--max_deg', type=int, required=True)
    p.add_argument('--step_deg', type=int, required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()

def extract_last_json(text: str):
    # robust: scan backwards for last '{' and try to load
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

def extract_e_day_wh(stdout: str):
    # Prefer proper JSON parse
    obj = extract_last_json(stdout)
    if isinstance(obj, dict) and 'e_day_wh' in obj:
        return float(obj['e_day_wh'])
    # fallback: regex if JSON had issues
    m = re.search(r'"e_day_wh"\s*:\s*([0-9]+(?:\.[0-9]+)?)', stdout)
    if m:
        return float(m.group(1))
    return None

def run_one_angle(runner, site, weather, date, deg):
    cmd = (
        f"python {runner} "
        f"--site {site} --weather {weather} --date {date} "
        f"--min_deg {deg} --max_deg {deg} --step_deg 1 "
        f"--offset_deg {deg}"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"[WARN] runner failed ({res.returncode}) at deg={deg}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
        return None
    e_wh = extract_e_day_wh(res.stdout)
    if e_wh is None:
        sys.stderr.write(f"[WARN] could not extract e_day_wh at deg={deg}\n")
        return None
    return e_wh

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for deg in range(args.min_deg, args.max_deg + 1, args.step_deg):
        e_wh = run_one_angle(args.runner, args.site, args.weather, args.date, deg)
        if e_wh is not None:
            rows.append((deg, e_wh / 1000.0))  # kWh
    if not rows:
        sys.stderr.write("[FATAL] No rows produced; aborting.\n")
        sys.exit(2)
    rows.sort(key=lambda t: t[0])
    with open(args.out, 'w') as f:
        f.write("deg,E_kWh\n")
        for deg, e in rows:
            f.write(f"{deg},{e}\n")
    print(json.dumps({"status":"ok","out":args.out,"n":len(rows)}))
    return

if __name__ == "__main__":
    main()
