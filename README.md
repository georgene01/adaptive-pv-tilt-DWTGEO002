# Adaptive PV Tilt – Results and Simulation Files  
<p align="left">
  <img src="https://img.shields.io/badge/Reproducible-Yes-brightgreen" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/pvlib-Used-informational" />
</p>

**Project:** Optimizing Photovoltaic Output Through Adaptive Tilt Angles to Mitigate Thermal Losses  
**Student:** Georgene de Wet (DWTGEO002)  
**Supervisor:** Prof K. A. Folly  
**Department:** Electrical Engineering, University of Cape Town  
**Course:** EEE4022 Final Year Project (2025)

---

## 1  Overview
This repository contains all scripts, configuration files, and processed data used to generate the results presented in the accompanying thesis report.  
The work investigates whether small, deliberate deviations from standard single-axis tracking angles can reduce PV module temperature enough to increase net AC energy yield.  

The simulation pipeline couples:
- NASA POWER and PVDAQ environmental datasets  
- The Faiman module-temperature model  
- PVLib-based DC/AC conversion and clipping calculations  
- Monte-Carlo LCOE estimation for annual and bin-wise tilt strategies  

All results, figures, and tables cited in the thesis (e.g., annual offset sweeps, bin energy comparisons, and LCOE analyses) were produced directly from the scripts and data contained here.

---

## 2  Folder Structure

```text
adaptive-pv-tilt-DWTGEO002/
├─ bins/           # Classified weather/irradiance bins for daily analysis
├─ configs/        # YAML configs for run scopes (e.g., run_scope.yaml)
├─ data_raw/       # Raw/aligned NASA POWER & PVDAQ inputs (large; often git-ignored)
├─ figs/           # Generated plots used in the report
├─ hourlies/       # Hourly POA / AC time-series exports
├─ inputs/         # Site lists, run inputs (e.g., sites.csv)
├─ manifests/      # Per-site/year manifests of aligned datasets
├─ out/            # Daily & annual energy summaries (lightweight)
├─ outputs/        # Heavier scratch outputs (often git-ignored)
├─ poa_col/        # POA irradiance collections/exports used by plots
├─ pv-tilt/        # Project-specific helpers or notebooks (if present)
├─ qc/             # Quality-control logs and patch commands
├─ selection/      # Selected representative days per bin and site
├─ tables/         # CSV/PDF exports cited in the thesis (e.g., LCOE priors, bin summaries)
├─ thesis/         # LaTeX report (if included)
├─ tmp/            # Temporary files / caches (git-ignored)
├─ tools/          # Simulation & plotting scripts (Python)
│  ├─ run_annual_policy.py
│  ├─ daily_sweep_direct.py
│  ├─ mc_lcoe_from_energy.py
│  ├─ plot_offtilt_results.py
│  └─ ... (other helper scripts)
├─ README.md
└─ README_patch_commands.txt

```
---

## 3  Reproducibility
All major results can be regenerated using Python ≥ 3.9 and the packages listed below.  

### 3.1  Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pvlib pandas numpy matplotlib scipy pyyaml

# Annual offset sweep and gain comparison (2023 vs 2024)
python tools/run_annual_policy.py

# Daily bin sweeps and energy attribution
python tools/daily_sweep_direct.py

# Monte-Carlo LCOE analysis
python tools/mc_lcoe_from_energy.py



# Re-generate plots used in thesis
python tools/plot_offtilt_results.py
```


```markdown
### Examiner quick links (verification)
- Annual offset sweeps (PDF):
  - [`tables/annual_energy_vs_offset_WC_2023.pdf`](tables/annual_energy_vs_offset_WC_2023.pdf)
  - [`tables/annual_energy_vs_offset_NC_2023.pdf`](tables/annual_energy_vs_offset_NC_2023.pdf)
  - [`tables/annual_energy_vs_offset_WC_2024.pdf`](tables/annual_energy_vs_offset_WC_2024.pdf)
  - [`tables/annual_energy_vs_offset_NC_2024.pdf`](tables/annual_energy_vs_offset_NC_2024.pdf)
- Bin attribution & gains:
  - [`tables/bin_data_results/`](tables/bin_data_results/)
  - [`tables/expected_gain_by_month_WC.csv`](tables/expected_gain_by_month_WC.csv)
  - [`tables/expected_gain_by_month_NC.csv`](tables/expected_gain_by_month_NC.csv)
  - [`tables/expected_gain_by_season_WC.csv`](tables/expected_gain_by_season_WC.csv)
  - [`tables/expected_gain_by_season_NC.csv`](tables/expected_gain_by_season_NC.csv)
- Daily energy audit:
  - [`tables/daily_energy_all.csv`](tables/daily_energy_all.csv)
```

## 4  Data Sources


| Dataset | Source | Description |
|----------|---------|-------------|
| `*_POWER_qc.csv` | NASA POWER API | Hourly irradiance, temperature, and wind data (quality-checked) |
| `PVDAQ1430` | NREL PVDAQ Database | Reference dataset for model validation |
| `*_manifest.csv` | Local | Site-year pairing of aligned input files |

All third-party datasets are open-source and publicly accessible.  
No personally identifiable or proprietary data are included.

---

## 6  License

MIT License  

Copyright (c) 2025 **Georgene de Wet**

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the “Software”), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.

---

## 7  Notes for Examiners

- All figures, tables, and quantitative results presented in the thesis were produced directly from the scripts and data contained in this repository.  
- The repository is structured to allow full reproducibility of the simulation pipeline described in the thesis.  
- Randomized elements within Monte-Carlo routines are seeded for deterministic replication.  
- The `tables/` directory contains all CSV exports that were imported into LaTeX for inclusion in the report.  
- Large raw files (e.g., NASA POWER `.csv` data, PVDAQ reference datasets) have been intentionally excluded from the public GitHub version to reduce size, but filenames and manifests are retained for full traceability.  
- The link to this GitHub repository is included in **Appendix C** of the thesis under *“Repository and Reproducibility”*.  
- The project satisfies the **Graduate Attributes GA 4, 5, 6, 8 and 9** through original data processing, simulation design, and professional presentation of technical communication.  
- Repository maintained by **Georgene de Wet (DWTGEO002)**, University of Cape Town, 2025.


