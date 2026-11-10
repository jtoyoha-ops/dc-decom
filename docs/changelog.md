# Changelog

All notable changes to DC-DECOM will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] — 2026-11-10

Version 2.0.0 is the updated model (DC-DECOM v2) of the doctoral thesis of Jin Toyohara (Graduate School of Policy Science, Ritsumeikan University). Its results differ from those of 1.x and from the values reported in the *Energies* article (2026, 19(10), 2485). No 1.0.2 release was made; the fixes of the v1 package are part of 2.0.0.

### Fixed (v1 package `src/dc_decom`)
Implementation errors, i.e. code that contradicted an explicit statement of the article (Section 3, Tables 7–9):
- **E1** `Scenario.share_2040` is a mapping by region name. In 1.0.1 it was a tuple ordered (Kanto, Kansai, Hokkaido, Tohoku, Kyushu) that the model applied in the row order of `regions.csv` (Hokkaido, Tohoku, Kanto, Kansai, Kyushu), so the R-CN 2040 targets went to the wrong regions (Table 7, Section 4.5).
- **E2** BESS discharge requires charging: the energy balance includes the charging term and `E_disch = bess_efficiency × E_ch` (Eq. 12). In 1.0.1 the BESS discharged without charging.
- **E3** Resilience backup `Cap_CGS ≥ (1 + 1/N) × IT load` with N = 1 (Eq. 16, Table 8). In 1.0.1 it was relaxed to 0.05 × IT load.
- **E4** 2024–2025 are calibration years (Section 3.9): no DLC, immersion, PV or BESS additions, regional shares at the base-year values, R-CN shares moving linearly from 2025 to 2040.
- **E5** Regional grid emission factors of Table 9 (Eq. 10) in the CO2 cap, the carbon cost and the reported emissions, declining in proportion to the national path. In 1.0.1 the national factor was used.
- **E6** `data/parameters_csv/regions.csv` follows Table 9 and Section 3.2: RE potential Tohoku 142 GW (was 180), Kansai 73 (75), Kyushu 115 (140), Hokkaido 337.471 (337), Kanto 98.833 (98.8); Kyushu PPA 13.0 yen/kWh (14.0); new column `grid_ef_base` (kg-CO2/kWh: Hokkaido 0.49, Tohoku 0.52, Kanto 0.44, Kansai 0.36, Kyushu 0.37).
- **E7** Eq. 15 bounds PV plus PPA capacity (`Cap_PV + E_PPA / (CF × 8.76) ≤ RE_max`). In 1.0.1 only PV was bounded.
- **E8** The R-CN flag `Scenario.regional_subsidy` (Table 7, regional DC subsidies) is applied: 5% discount on the CAPEX of DLC, immersion, CGS and BESS outside Kanto/Kansai, reported as `SolveResult.subsidy_total`. In 1.0.1 the flag was unused.

LP results of the v1 package with the default inputs (`src/scripts/run_all.py`): REF 15.59, C-CN 123.17, R-CN 113.18 trillion JPY (1.0.1: 10.60 / 11.37 / 11.75); R-CN advantage over C-CN +8.11% (1.0.1: −3.30%); cumulative curtailment C-CN 308.0 TWh, R-CN 279.6 TWh (1.0.1: 0). The reported values of the article (6.2 / 8.5 / 6.8 trillion JPY, 18–23%, Table 10 split 35/30/25/10) are not reproduced.

### Removed
- `dc_decom.postprocess.calibrated_headline()`, which returned the reported values of the article as constants next to the LP results.

### Changed (v1 package)
- `postprocess`: new `headline_lp()` (LP totals, advantage, curtailment, subsidy and a decomposition of the C-CN − R-CN difference whose five lines add up to the difference) and `reported_values()` (the reported values of the article, labelled as constants from the article, not model output). `cost_components()` adds the subsidy and curtailment columns.
- `SolveResult` has the new fields `subsidy_total`, `curtail_TWh_total` and `components` (discounted cost components per region and year); table rows have `E_ch_TWh`, `curt_TWh` and `ef_grid`. CBC is called with `threads=1`.
- `src/scripts/run_all.py` prints the LP results first and the reported values under the heading `Reported values (Energies 2026) - constants from the article, not model output`; `out/*.json` include the subsidy and the curtailment.
- `tests/test_headline_numbers.py`: the test that compared the reported constants with themselves is replaced by tests of the LP direction and curtailment and of the separation of the reported values from the LP output.
- `dc_decom.__version__` = "2.0.0".

### Added
- `src/dc_decom_v2/`: DC-DECOM v2 — annual and 96-representative-hour modes in one LP builder, BESS with state of charge, N+1 backup by CGS and emergency generators, endogenous siting with metro and Kanto floors, grid connection constraints with reinforcement cost, a stepped PPA supply curve, non-fossil certificates, annual or hourly matching, regional or national grid emission factors, social resource cost / Eq. 1 total / operator cost / fiscal burden accounting, rolling execution (plan → realize → observe) of the A/B/C designs, and a parallel cached runner.
- `src/dc_decom_v2/experiments/`: base case, bridge from the v2 base along two assumption sets (S1: assumptions stated in the article; S2: assumptions whose values are not stated in the article), conditions analysis, tornado, certificates, carbon-price sweep, P*, A/B/C comparison on 18 realization paths, interaction factorial, composite uncertainty (Latin hypercube Monte Carlo), breakeven of non-metro adders, training / inference load ratio, annual vs 96 hours, carbon price × certificate rule, Shapley decomposition of the emission reduction, structure × parameters 2×2, post-hoc sensitivity of the A/B/C rules and the breakdown of B − A0.
- `src/dc_decom_spec/`: the spec version (the v1 model with switchable fixes E1–E8 and readings of points the article does not state), used for the reference points of the bridge, the 2×2 and the breakeven (`src/scripts/run_spec.py` → `outputs/stage2/`).
- `src/dc_decom_analysis/`: principled cost decomposition, LMDI, Shapley values, Latin hypercube sampling with PRCC, output manifest.
- `config/`: experiment definitions (`experiments_v2.yaml`, `experiments_v35.yaml`, `interaction_pairs.yaml`, `abc_posthoc.yaml`) and the A/B/C rules (`abc_rules.yaml`, checked against its sha256 before every A/B/C run).
- `data/parameters_v2/`: v2 parameters with `sources.csv`, `SOURCES.md` and `FREEZE.md` (list of assumptions with ranges); `data/parameters_spec/`: inputs of the spec version; preprocessing scripts `data/preprocess_jma.py`, `data/calibrate_pue_v2.py`, `data/build_parameters_v2.py`; retrieval records of the raw data in `data/raw/` (the third-party files themselves are not included).
- `outputs/stage2/`, `outputs/stage3/`, `outputs/abc/`, `outputs/manifest.json`: the committed results of `src/scripts/run_thesis_v2.py --all`.
- `src/scripts/run_thesis_v2.py` (one command for every output), `src/scripts/run_stage3.py` (one part at a time, `--check` against the committed files), `src/scripts/run_spec.py`.
- Tests of the v2 model, the spec version and the analysis helpers (160 tests in total).

### Changed (repository)
- CI (`.github/workflows/reproducibility.yml`): ruff on `src/` and `tests/`, mypy on the four packages, the v1 scenarios, the spec runs and the full test suite. The apt installation of CBC is dropped (the code uses the CBC bundled with PuLP).
- `README.md`, `CITATION.cff` (version 2.0.0; the doctoral thesis as preferred citation; concept DOI), `LICENSE-DATA` (attribution of the JMA, MOE/METI and EGC data).

## [1.0.1] — 2026-05-22
- Journal article published: *Energies* 2026, *19*(10), 2485. DOI [10.3390/en19102485](https://doi.org/10.3390/en19102485).
- `CITATION.cff` and `README.md` updated with the journal article DOI; software archive Zenodo DOI unchanged.

## [1.0.0] — 2026-05-21
- Initial public release. LP formulation, scenario drivers (REF / C-CN / R-CN), input parameter spreadsheet (mirrors paper Table 8), 96-hour temperature profiles for 5 regions, and reproducibility unit tests (14 passing).
- Released under personal GitHub account [@jtoyoha-ops](https://github.com/jtoyoha-ops) (not under any Organization).
- Code: MIT. Data: CC BY 4.0.
- Zenodo DOI: [10.5281/zenodo.20312805](https://doi.org/10.5281/zenodo.20312805)
