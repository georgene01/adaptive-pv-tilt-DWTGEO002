# Tools for Results Step 1.4 (POWER fetch + QC)

## Fetch
python tools/power_fetch.py --lat -28.5 --lon 21.0 --start 2024-01-01 --end 2024-12-31 --out data_raw/NC_2024_POWER.csv
python tools/power_fetch.py --lat -33.9 --lon 18.6 --start 2024-01-01 --end 2024-12-31 --out data_raw/WC_2024_POWER.csv

## Quality Control
python tools/qc_weather.py --in data_raw/NC_2024_POWER.csv --out qc/NC_2024_qc_report.json --timeshift qc/NC_2024_timeshift_check.txt
python tools/qc_weather.py --in data_raw/WC_2024_POWER.csv --out qc/WC_2024_qc_report.json --timeshift qc/WC_2024_timeshift_check.txt

Outputs:
  - data_raw/*_2024_POWER_qc.csv   (cleaned, ready for modelling)
  - qc/*_2024_qc_report.json       (counts: drops/clamps/bounds/interpolations)
  - qc/*_2024_timeshift_check.txt  (checklist notes for UTC->SAST sanity)
