# NASA POWER hourly fetcher (hourly-safe variables only).
# Usage (run as a single line):
#   python tools/power_fetch.py --lat -28.5 --lon 21.0 --start 2024-01-01 --end 2024-12-31 --out data_raw/NC_2024_POWER.csv
#
# Notes:
# - We request only vars the hourly endpoint serves reliably to avoid 422.
# - We'll compute any extra diagnostics downstream.

import argparse, csv, datetime as dt, requests

# Hourly-safe list (drop CLRSKY_*, DNR, DIF, ALLSKY_SFC_LW_DWN)
VARS = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "RH2M", "PS", "ALLSKY_SRF_ALB"]

def fetch_hourly(lat, lon, start, end):
    base = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": ",".join(VARS),
        "community": "RE",
        "longitude": lon,
        "latitude":  lat,
        "start":     start.strftime("%Y%m%d"),
        "end":       end.strftime("%Y%m%d"),
        "format":    "JSON",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def flatten(js):
    props = js.get("properties", {})
    param = props.get("parameter", {})
    # collect all time keys present across vars
    timekeys = set()
    for v in param.values():
        timekeys |= set(v.keys())

    rows = []
    for tk in sorted(timekeys):
        key = tk.replace(":", "")
        if len(key) == 10:   # 2024010101
            ts = dt.datetime.strptime(key, "%Y%m%d%H")
        elif len(key) == 11: # 20240101:01 (rare)
            ts = dt.datetime.strptime(key, "%Y%m%d:%H")
        else:
            continue
        rows.append({
            "time_utc":         ts.strftime("%Y-%m-%d %H:00"),
            "ghi_wm2":          param.get("ALLSKY_SFC_SW_DWN", {}).get(tk),
            "temp_air_c":       param.get("T2M", {}).get(tk),
            "wind_speed_ms":    param.get("WS10M", {}).get(tk),
            "rel_humidity_pct": param.get("RH2M", {}).get(tk),
            "pressure_pa":      (param.get("PS", {}).get(tk) * 100) if param.get("PS", {}).get(tk) is not None else None,
            "albedo":           param.get("ALLSKY_SRF_ALB", {}).get(tk),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)  # YYYY-MM-DD
    ap.add_argument("--end",   type=str, required=True)  # YYYY-MM-DD
    ap.add_argument("--out",   type=str, required=True)
    args = ap.parse_args()

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()

    js   = fetch_hourly(args.lat, args.lon, start, end)
    rows = flatten(js)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time_utc","ghi_wm2","temp_air_c","wind_speed_ms","rel_humidity_pct","pressure_pa","albedo"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote: {args.out}  (rows={len(rows)})")

if __name__ == "__main__":
    main()
