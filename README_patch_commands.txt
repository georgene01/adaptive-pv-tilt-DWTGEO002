# Zsh-safe one-liners to re-run the pipeline (copy each line separately)

# Fetch NC
python tools/power_fetch.py --lat -28.5 --lon 21.0 --start 2024-01-01 --end 2024-12-31 --out data_raw/NC_2024_POWER.csv
# QC NC
python tools/qc_weather.py --in data_raw/NC_2024_POWER.csv --out qc/NC_2024_qc_report.json --timeshift qc/NC_2024_timeshift_check.txt
# Fetch WC
python tools/power_fetch.py --lat -33.9 --lon 18.6 --start 2024-01-01 --end 2024-12-31 --out data_raw/WC_2024_POWER.csv
# QC WC
python tools/qc_weather.py --in data_raw/WC_2024_POWER.csv --out qc/WC_2024_qc_report.json --timeshift qc/WC_2024_timeshift_check.txt

# Select + render tables
python tools/select_bins.py --site NC --in data_raw/NC_2024_POWER_qc.csv --out selection/NC_2024_day_bins.csv --fig figs/day_bin_table.pdf
python tools/select_bins.py --site WC --in data_raw/WC_2024_POWER_qc.csv --out selection/WC_2024_day_bins.csv --fig figs/day_bin_table_WC.pdf
