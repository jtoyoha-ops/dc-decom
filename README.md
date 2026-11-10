# DC-DECOM

[![CI](https://github.com/jtoyoha-ops/dc-decom/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/jtoyoha-ops/dc-decom/actions/workflows/reproducibility.yml)
[![Article DOI](https://img.shields.io/badge/Article%20DOI-10.3390%2Fen19102485-blue.svg)](https://doi.org/10.3390/en19102485)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20312804.svg)](https://doi.org/10.5281/zenodo.20312804)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Data CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**DC-DECOM: a bottom-up cost-optimization model for data center decarbonization in Japan**

- **v2.0.0** (this version) is the updated model, DC-DECOM v2, of the doctoral thesis of Jin Toyohara (Graduate School of Policy Science, Ritsumeikan University).
- **v1.x** accompanies Toyohara & Zhou (2026), *Energies* 19(10), 2485 ([10.3390/en19102485](https://doi.org/10.3390/en19102485)).
- The results of v2.0.0 differ from the results of v1.x and from the values reported in the article (table below).

> Published under the personal GitHub account [@jtoyoha-ops](https://github.com/jtoyoha-ops). This is an individual research artefact; the repository is **not** released under any Organization that the account is a member of, and no Organization endorsement is implied.

---

## Main changes in v2.0.0

1. **Fixes of the v1 package** (`src/dc_decom`): implementation errors E1–E8, i.e. code that contradicted Section 3 and Tables 7–9 of the article (regional order of the 2040 shares, BESS charging, N+1 backup, calibration years, regional emission factors, Table 9 inputs, Eq. 15, R-CN subsidy). The function `calibrated_headline()`, which returned the reported values of the article as constants, is removed. The reported values are available only through `reported_values()`, labelled as constants from the article, not model output. Details: [docs/changelog.md](docs/changelog.md).
2. **96 representative hours** (4 seasons × 24 hours) in one LP builder that also has an annual mode (`src/dc_decom_v2/model.py`).
3. **Endogenous siting**: the LP allocates national IT demand to the five regions, with a metro floor λ, a Kanto floor κ and no relocation of existing load.
4. **Grid constraints and reinforcement cost**: connection capacity for new non-metro load, reinforcement from a given year at a cost per kW of connected peak.
5. **Non-fossil certificates** as a procurement option, next to a stepped PPA supply curve and on-site PV; annual (market-based) or hourly (24/7) matching.
6. **PUE recalibration with observed temperatures**: the air-cooling PUE coefficients are refitted with the hourly temperatures of the Japan Meteorological Agency (JMA) at Sapporo, Sendai, Tokyo, Osaka and Fukuoka, 2020–2023 (`data/calibrate_pue_v2.py`, `data/parameters_v2/pue_calibration.json`).
7. **Rolling execution and the A/B/C comparison**: plan → realize → observe at the review years 2026, 2030 and 2035 on 18 realization paths of carbon price, demand and liquid-cooling deployment (`src/dc_decom_v2/rolling.py`, `config/abc_rules.yaml`).
8. **Uncertainty analysis**: Latin hypercube Monte Carlo (three blocks of 1,000 samples), breakeven of non-metro cost adders, tornado and carbon-price sweeps, Shapley decomposition of the emission reduction, structure × parameters 2×2, post-hoc sensitivity of the A/B/C rules.

Other parts of v2: BESS with state of charge, N+1 backup by CGS and emergency generators, regional grid emission factors (SHK residual factors, FY2024), a parameter set in which every value carries a source or is marked as an assumption with a range (`data/parameters_v2/`).

## Results of the versions compared

Trillion JPY, NPV 2024–2040, discount rate 3%. Advantage = (C-CN − R-CN) / C-CN.

| | REF | C-CN | R-CN | R-CN advantage | Source |
|---|---|---|---|---|---|
| v2.0.0, DC-DECOM v2 base case | 9.37 | 9.37 | 8.82 | 5.9% | `outputs/stage3/base_summary.json` |
| v2.0.0, v1 package with E1–E8 fixed | 15.59 | 123.17 | 113.18 | 8.1% | `python src/scripts/run_all.py` |
| v1.0.1, v1 package | 10.60 | 11.37 | 11.75 | −3.3% | `src/scripts/run_all.py` at tag v1.0.1 (LP output) |
| Reported values of Toyohara & Zhou (2026), not model output | 6.2 | 8.5 | 6.8 | 18–23% | *Energies* Section 4.2, Table 10 |

- v2 totals are the Eq. 1 total of the thesis: social resource cost plus carbon cost; subsidies are transfers and are not counted (`config/experiments_v2.yaml`, `metrics`). There is no curtailment in the v2 base case.
- v1 totals are the LP objective of the v1 package: it includes the curtailment penalty (500 yen/kWh) and deducts the R-CN subsidy. With E1–E8 fixed, C-CN and R-CN meet the CO2 cap only with curtailment (308.0 and 279.6 TWh, cumulative), and the penalty makes up most of their totals.

---

## Reproducing the results

Requirements:

- Python 3.11 and the pinned packages of `requirements.txt` (PuLP 2.8.0 with its bundled CBC 2.10.3; no separate solver installation).
- About 3.6 GB of free disk space for the solve cache `outputs/cache/` (not committed).

```bash
git clone https://github.com/jtoyoha-ops/dc-decom.git
cd dc-decom
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q                   # 160 tests, about 1 minute
python src/scripts/run_thesis_v2.py --all    # regenerates every file under outputs/
git status --short outputs/                  # committed files that changed
```

`run_thesis_v2.py --all` runs the spec version (`src/scripts/run_spec.py` → `outputs/stage2/`), every part of `src/scripts/run_stage3.py` (→ `outputs/stage3/`, `outputs/abc/`) and writes `outputs/manifest.json` (git commit, package and CBC versions, run time per part, sha256 of every output file).

Time on the development machine (macOS on Apple silicon, the x86_64 CBC of PuLP under Rosetta 2, 8 worker processes): about 6 hours when every LP is solved, of which the Monte Carlo part (`mc`) takes about 5 hours (measured on 2026-11-09); about 4 minutes when the solves are read from `outputs/cache/`.

- A single part: `python src/scripts/run_stage3.py --part <name>`; `--check` recomputes the part into a temporary directory and compares every file with the committed one.
- CBC runs with one thread. On the development machine, re-solving every LP of every part without the cache (2026-11-09) gave value files (CSV, JSON, Markdown) byte-identical to the earlier runs. `run_log_*.csv` contain solve times and differ after re-solving; `manifest.json` differs in its time stamps and git state. Other platforms and CBC builds have not been checked.
- The v1 model alone: `python src/scripts/run_all.py` (REF, C-CN, R-CN; under a minute).

## Repository layout

```
dc-decom/
├── src/
│   ├── dc_decom/            # v1 model (Energies 2026), E1-E8 fixed
│   ├── dc_decom_spec/       # spec version: v1 with switchable fixes E1-E8 and readings
│   ├── dc_decom_v2/         # DC-DECOM v2 (model, parameters, rolling execution, runner)
│   │   └── experiments/     # base, bridge, sensitivity, A/B/C, uncertainty, ...
│   ├── dc_decom_analysis/   # decomposition, LMDI, Shapley values, LHS/PRCC, manifest
│   └── scripts/             # run_thesis_v2.py, run_stage3.py, run_spec.py, run_all.py, run_*.py
├── config/                  # experiment definitions and the A/B/C rules
├── data/
│   ├── parameters_csv/      # inputs of the v1 package
│   ├── parameters_spec/     # inputs of the spec version (Energies Tables 6, 8, 9)
│   ├── parameters_v2/       # v2 parameters, sources.csv, SOURCES.md, FREEZE.md
│   ├── raw/                 # retrieval records of the raw data (files not included)
│   └── *.py                 # preprocessing (JMA profiles, PUE recalibration, parameter build)
├── outputs/
│   ├── stage2/              # spec version (reference points of the bridge and the 2x2)
│   ├── stage3/              # v2 base and analyses (figure data of the thesis included)
│   ├── abc/                 # A/B/C comparison, interaction factorial, post-hoc sensitivity
│   └── manifest.json
├── tests/
└── docs/changelog.md
```

The experiment definitions (`config/`) and the v2 parameters (`data/parameters_v2/`) were fixed before the runs that use them; later changes are listed in the files (`changes_after_registration`). `config/abc_rules.yaml` is checked against its sha256 before every A/B/C run.

## Data sources and terms of use

- `data/parameters_v2/sources.csv` gives the bibliographic record, URL and access date of every source; `SOURCES.md` (in Japanese) gives the derivations. Each value has a status: `sourced`, `derived`, `paper` (a table of the Energies article) or `assumption` (no source; low/high range used in the sensitivity analysis).
- Third-party original files (JMA hourly CSV files, the monthly xlsx files of the Electricity and Gas Market Surveillance Commission, the emission-factor xlsx of the Ministry of the Environment) are not included. `data/raw/*/` records the URL, the access date and the sha256 of each file; `data/raw/jma/fetch_jma.py` downloads the JMA files.
- Attribution as required by the terms of use of the sources (Public Data License 1.0 of the Japanese government websites):
  - 出典：気象庁ホームページ（https://www.data.jma.go.jp/risk/obsdl/）の時別値（2020〜2023年、札幌・仙台・東京・大阪・福岡の気温・全天日射量）を加工して作成。Source: Japan Meteorological Agency website, hourly temperature and global solar radiation, processed by the author (`data/parameters_v2/temperature_96h_jma.csv`, `pv_cf_96h_jma.csv`, `jma_summary.csv` and the values derived from them).
  - 環境省・経済産業省「電気事業者別排出係数（特定排出者の温室効果ガス排出量算定用）－R6年度実績－」（https://policies.env.go.jp/earth/ghg-santeikohyo/calc.html）を加工して作成（`data/parameters_v2/ef_fy2024.csv`）。
  - 電力・ガス取引監視等委員会「電力取引の状況（電力取引報）」（https://www.egc.meti.go.jp/info/business/report/results.html）を加工して作成（`data/parameters_v2/tariff_fy2025.csv`）。
  - The processed data were made by the author and are not products of these agencies.

## Licensing

- **Code**: MIT License, see `LICENSE`. Copyright (c) 2026 Jin Toyohara.
- **Data**: CC BY 4.0, see `LICENSE-DATA`. The attribution requirements of the original sources (above) apply in addition.

## How to cite

`CITATION.cff` gives the machine-readable citation (GitHub shows it as "Cite this repository").

- v2.0.0 (DC-DECOM v2): Toyohara, J. *Policy Engineering for Data Center Decarbonization in the Era of Generative AI: Integrating Technology Assessment, System Optimization, and International Institutional Comparison* (生成AI時代のデータセンター脱炭素化に関する政策工学研究―技術評価・システム最適化・国際制度比較の統合―). Doctoral dissertation, Graduate School of Policy Science, Ritsumeikan University, 2027.
- v1.x: Toyohara, J.; Zhou, W. *Cost-Optimal Decarbonization Pathways for Data Centers in Japan: A Bottom-Up Model Integrating Location, Energy Systems, and Carbon Pricing*. **Energies** 2026, *19*(10), 2485. [https://doi.org/10.3390/en19102485](https://doi.org/10.3390/en19102485)
- Software archive (Zenodo): concept DOI [10.5281/zenodo.20312804](https://doi.org/10.5281/zenodo.20312804) (all versions; resolves to the latest). The article's Data Availability statement cites the v1.0.0 version DOI [10.5281/zenodo.20312805](https://doi.org/10.5281/zenodo.20312805). Zenodo assigns a version DOI to v2.0.0 when the release is published.

---

## Contact and contribution

- Maintainer: Jin Toyohara — via GitHub Issues on this repository.
- Pull requests are welcome for documentation improvements and reproducibility fixes.
- Issue/PR triage target: within 30 days during the first two years post-publication.
