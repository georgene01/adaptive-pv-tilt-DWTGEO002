#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive daily sweep runner.
- Introspects sim_support.run_daily signature and calls it with matching args.
- Writes CSV: tmp/daily_sweep/{site}_{date}.csv -> columns: deg,E_kWh
CLI:
  --site NC --weather tmp/daily_weather/NC/NC_2024-01-01.csv --date 2024-01-01 \
  --min_deg -14 --max_deg 14 --step_deg 2 [--out tmp/daily_sweep/NC_2024-01-01.csv]
"""
import argparse, os, sys, json, inspect

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--site', required=True)
    p.add_argument('--weather', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--min_deg', type=int, required=True)
    p.add_argument('--max_deg', type=int, required=True)
    p.add_argument('--step_deg', type=int, required=True)
    p.add_argument('--out', default=None)
    return p.parse_args()

def build_args_for_run_daily(fn, site, weather, date, min_deg, max_deg, step_deg):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    have = {
        'site': site,
        'weather': weather,
        'weather_path': weather,
        'meteo': weather,
        'date': date,
        'day': date,
        'min_deg': min_deg,
        'min_angle': min_deg,
        'max_deg': max_deg,
        'max_angle': max_deg,
        'step_deg': step_deg,
        'deg_step': step_deg,
        'step': step_deg,
        # angle lists (some APIs want a list instead of min/max/step)
        'angles': list(range(min_deg, max_deg+1, step_deg)),
        'deg_list': list(range(min_deg, max_deg+1, step_deg)),
        'angle_list': list(range(min_deg, max_deg+1, step_deg)),
    }

    # Try kwargs by name where possible
    kwargs = {}
    for p in params:
        name = p.name
        if name in have:
            kwargs[name] = have[name]

    # If the function accepts **kwargs, kwargs is fine. If not, build positionals in order.
    try:
        return (), kwargs, sig
    except Exception:
        pass

    # Build ordered positional args in the function's declared order using whatever we matched.
    pos = []
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if p.name in have:
                pos.append(have[p.name])
            elif p.default is not p.empty:
                pos.append(p.default)
            else:
                raise TypeError(f"Don't know how to supply param '{p.name}' for run_daily")
    return tuple(pos), {}, sig

def rows_from_any(res):
    """Normalize run_daily return into [{'deg':..,'E_kWh':..},...]"""
    def rows_from_list(lst):
        if not lst or not isinstance(lst[0], dict): return None
        keys = set().union(*[d.keys() for d in lst])
        def pick(cands):
            for k in cands:
                if k in keys: return k
            kl = {k.lower():k for k in keys}
            for k in cands:
                if k in kl: return kl[k]
            return None
        k_deg = pick(['deg','angle','theta','offset_deg','off_deg'])
        k_E   = pick(['E_kWh','energy_kWh','energy','E','Eday_kWh','E_day_kWh'])
        if not k_deg or not k_E: return None
        out = []
        for d in lst:
            if k_deg in d and k_E in d:
                out.append({'deg': d[k_deg], 'E_kWh': d[k_E]})
        return out if out else None

    if isinstance(res, list):
        r = rows_from_list(res)
        if r: return r
    if isinstance(res, dict):
        for k in ('curve','results','data'):
            if k in res and isinstance(res[k], list):
                r = rows_from_list(res[k])
                if r: return r
        # fallback: dict mapping angle->energy
        if res and all(isinstance(k, (int,float)) for k in res.keys()):
            return [{'deg': int(k), 'E_kWh': float(v)} for k,v in sorted(res.items())]
        # search nested
        for v in res.values():
            r = rows_from_any(v)
            if r: return r
    return None

def main():
    args = parse_args()
    out_csv = args.out or f"tmp/daily_sweep/{args.site}_{args.date}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    from sim_support import run_daily

    # Try: kwargs first (by name), else positionals
    pos, kw, sig = build_args_for_run_daily(
        run_daily, args.site, args.weather, args.date, args.min_deg, args.max_deg, args.step_deg
    )
    try:
        if kw:
            res = run_daily(**kw)
        else:
            res = run_daily(*pos)
    except TypeError as e1:
        # If first attempt failed, try the alternate path (positional vs kwargs)
        try:
            if kw:
                # convert to positional in function's order
                params = list(sig.parameters.values())
                have = {
                    'site': args.site, 'weather': args.weather, 'weather_path': args.weather, 'meteo': args.weather,
                    'date': args.date, 'day': args.date,
                    'min_deg': args.min_deg, 'min_angle': args.min_deg,
                    'max_deg': args.max_deg, 'max_angle': args.max_deg,
                    'step_deg': args.step_deg, 'deg_step': args.step_deg, 'step': args.step_deg,
                    'angles': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                    'deg_list': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                    'angle_list': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                }
                pos2 = []
                for p in params:
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                        if p.name in have: pos2.append(have[p.name])
                        elif p.default is not p.empty: pos2.append(p.default)
                        else: raise
                res = run_daily(*pos2)
            else:
                # try kwargs from positional mapping
                params = list(sig.parameters.values())
                have = {
                    'site': args.site, 'weather': args.weather, 'weather_path': args.weather, 'meteo': args.weather,
                    'date': args.date, 'day': args.date,
                    'min_deg': args.min_deg, 'min_angle': args.min_deg,
                    'max_deg': args.max_deg, 'max_angle': args.max_deg,
                    'step_deg': args.step_deg, 'deg_step': args.step_deg, 'step': args.step_deg,
                    'angles': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                    'deg_list': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                    'angle_list': list(range(args.min_deg, args.max_deg+1, args.step_deg)),
                }
                kw2 = { p.name: have[p.name] for p in params if p.name in have }
                res = run_daily(**kw2)
        except Exception as e2:
            print(f"[FATAL] run_daily invocation failed: {e1} | {e2}", file=sys.stderr)
            sys.exit(2)

    rows = rows_from_any(res)
    if not rows:
        print(f"[FATAL] Could not interpret run_daily() return; got type {type(res)}", file=sys.stderr)
        sys.exit(3)

    with open(out_csv, 'w') as f:
        f.write("deg,E_kWh\n")
        for r in rows:
            f.write(f"{int(r['deg'])},{float(r['E_kWh'])}\n")

    print(json.dumps({'status':'ok','out':out_csv,'n':len(rows)}))
    return

if __name__ == "__main__":
    main()
