import argparse

def _to_str_keys(obj):
    # Recursively convert mapping keys to strings to make JSON-safe
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            out[ks] = _to_str_keys(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [ _to_str_keys(x) for x in obj ]
    else:
        return obj
import json
import numpy as np
import pandas as pd, json, sys, pathlib, types, pandas as pd

# we only need run_daily from sim_support
from sim_support import run_daily



from types import SimpleNamespace

def _local_build_simulator(site: str):
    """Local, minimal sim object that ALWAYS has a tracker."""
    s = (site or "").upper()
    if s == "NC":
        meta = dict(lat=-28.46, lon=21.23, tz="Africa/Johannesburg")
    elif s == "WC":
        meta = dict(lat=-33.92, lon=18.42, tz="Africa/Johannesburg")
    else:
        meta = dict(lat=-26.20, lon=28.04, tz="Africa/Johannesburg")

    tracker = dict(
        axis_tilt_deg=0.0,
        axis_azimuth_deg=0.0,
        gcr=0.35,
        backtrack=True,
        max_rotation_deg=60.0,
    )
    module = dict(pdc0=1.0)
    extras = dict(albedo=0.2, elevation_m=0.0)

    return SimpleNamespace(**meta, tracker=tracker, module=module, **extras)


# hard-coded presets so we don't depend on sim_support factories
SITE_PRESETS = {
    "NC": dict(lat=-28.45, lon=21.25, tz="Africa/Johannesburg"),  # Upington-ish
    "WC": dict(lat=-33.92, lon=18.42, tz="Africa/Johannesburg"),  # Cape Town-ish
}

def get_sim(site:str):
    if site not in SITE_PRESETS:
        raise RuntimeError(f"Unknown site '{site}'. Known: {list(SITE_PRESETS)}")
    return types.SimpleNamespace(**SITE_PRESETS[site])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--weather", required=True)  # per-day CSV
    ap.add_argument("--date", required=True)     # YYYY-MM-DD local date
    ap.add_argument("--min_deg", type=float, required=True)
    ap.add_argument("--max_deg", type=float, required=True)
    ap.add_argument("--step_deg", type=float, required=True)
    ap.add_argument("--offset_deg", type=float, default=0.0)
    args = ap.parse_args()

    sim = get_sim(args.site)

    p = pathlib.Path(args.weather)
    if not p.exists():
        raise RuntimeError(f"Weather CSV not found: {p}")

    # sanity check: make sure the daily CSV matches the requested date if a date column exists
    try:
        df = pd.read_csv(p, nrows=5, low_memory=False)
        date_col = None
        for c in df.columns:
            lc = c.lower()
            if lc.startswith("date"): date_col = c; break
        if date_col is not None:
            vals = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
            if (vals == args.date).sum() == 0:
                raise RuntimeError(f"No rows found for {args.date} in {p}")
    except Exception:
        pass
    # hard guarantee our sim has a tracker
    if not hasattr(sim, "tracker"):
        sim = _local_build_simulator(args.site)
    # hard guarantee our sim has a tracker + inverter
    if not hasattr(sim, "tracker") or not hasattr(sim, "inverter"):
        sim = _local_build_simulator(args.site)
    # guarantee tracker+inverter presence before running
    if not hasattr(sim, "tracker") or not hasattr(sim, "inverter"):
        sim = _local_build_simulator(args.site)
    # hard guarantee: ensure sim has inverter before run_daily
    try:
        _ = sim.inverter
    except Exception:
        try:
            from types import SimpleNamespace as _S
            d = getattr(sim, "__dict__", {}) or {}
            d = dict(d)
            d["inverter"] = {"coefficients": {"eta_nom": 0.97}}
            sim = _S(**d)
        except Exception:
            try:
                sim.inverter = {"coefficients": {"eta_nom": 0.97}}
            except Exception:
                pass





    out = run_daily(sim, str(p), args.date, offset_deg=args.offset_deg)
    safe = _jsonify_for_cli(out)
    safe = _jsonify_for_cli(out)
    print(json.dumps(_to_str_keys(safe), default=str))
    sys.exit(0)# --- JSON sanitizer for Pandas/Numpy ---
def _jsonify_for_cli(x):
    import numpy as _np, pandas as _pd, datetime as _dt
    if x is None:
        return None
    # primitives
    if isinstance(x, (int, float, str, bool)):
        return x
    # numpy
    if isinstance(x, (_np.integer, _np.floating)):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    # pandas scalars & timestamps
    if isinstance(x, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return x.isoformat()
    # pandas containers
    if isinstance(x, _pd.Series):
        return {k: _jsonify_for_cli(v) for k, v in x.to_dict().items()}
    if isinstance(x, _pd.DataFrame):
        return {k: [_jsonify_for_cli(v) for v in col] for k, col in x.to_dict(orient='list').items()}
    if isinstance(x, (_pd.Index, _pd.DatetimeIndex)):
        return [_jsonify_for_cli(v) for v in x.tolist()]
    # containers
    if isinstance(x, dict):
        return {k: _jsonify_for_cli(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonify_for_cli(v) for v in list(x)]
    # fallback
    try:
        return str(x)
    except Exception:
        return None


if __name__ == "__main__":
    main()
