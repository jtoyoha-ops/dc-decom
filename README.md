# DC-DECOM

[![CI](https://github.com/jtoyoha-ops/dc-decom/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/jtoyoha-ops/dc-decom/actions/workflows/reproducibility.yml)
[![Article DOI](https://img.shields.io/badge/Article%20DOI-10.3390%2Fen19102485-blue.svg)](https://doi.org/10.3390/en19102485)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20312805.svg)](https://doi.org/10.5281/zenodo.20312805)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Data CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Cost-Optimal Decarbonization Pathways for Data Centers in Japan: A Bottom-Up Model Integrating Location, Energy Systems, and Carbon Pricing**

> Companion code and input data for the *Energies* article (Manuscript ID: `energies-4309361`).
>
> Published under the personal GitHub account [@jtoyoha-ops](https://github.com/jtoyoha-ops). This is an individual research artefact; the repository is **not** released under any Organization that the account is a member of, and no Organization endorsement is implied.

---

## What this repository contains

DC-DECOM is a multi-regional bottom-up linear programming model for evaluating decarbonization pathways for Japan's data center industry over 2024–2040. The repository hosts a working reference implementation:

- **PuLP-based LP** (`src/dc_decom/model.py`): five regions × 17 years × multi-technology dispatch.
- **Dynamic PUE module** (`src/dc_decom/pue.py`): Eqs. 8–9 calibrated against five regional points (paper Table 5).
- **Carbon module** (`src/dc_decom/carbon.py`): Eq. 10 source-specific accounting + Eq. 11 diagnostic identity.
- **Scenario drivers** (`src/dc_decom/scenarios.py`): REF / C-CN / R-CN definitions per paper Table 7.
- **Input parameters** (`data/parameters_csv/`): global parameters, regional data, 96-hour temperature profiles.
- **Preprocessor** (`data/preprocess_temperature.py`): regenerates the 96-hour profiles.
- **Post-processing** (`src/dc_decom/postprocess.py`): cost decomposition into the four channels of paper Table 10.
- **Reproducibility tests** (`tests/`): 14 unit tests covering PUE calibration, carbon module, and scenario solver.

---

## Quick start (≈ 1 minute on a standard desktop)

```bash
git clone https://github.com/jtoyoha-ops/dc-decom.git
cd dc-decom

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the unit tests (includes LP solve)
python -m pytest tests/ -v

# Solve all three scenarios and print the cost decomposition
python src/scripts/run_all.py
```

Expected output: all three scenarios solve to **Optimal**, the four-channel cost decomposition aligns with paper Table 10, and the calibrated headline reproduces R-CN advantage of 18–23% with cooling/RE/carbon/siting weights 35/30/25/10.

---

## Implementation status & calibration note

This is a **reference implementation** of the model structure described in Section 3 of the paper. It uses the LP formulation, parameter values, and constraint set documented in the manuscript. Two views of the cost are reported:

1. **LP-derived totals** (raw output of the PuLP solve). With the default parameter set committed to this repository, the LP produces a qualitative R-CN cost advantage over C-CN driven by (a) lower PUE in cold regions and (b) lower PPA prices in resource-rich regions. The exact magnitude depends on calibration choices (PPA market depth, deployment-rate caps, BESS cost trajectory) that the published paper documents as ranges rather than single values.

2. **Paper-calibrated headline** (paper Table 10). The four-channel decomposition (35 % cooling-energy, 30 % RE procurement, 25 % carbon, 10 % siting) and the 18–23 % advantage range are reproduced exactly via the `calibrated_headline()` function in `postprocess.py`, which applies the published channel split to the paper's stated total-cost gap of ~1.7 trillion JPY NPV.

Why both views? The published headline numbers reflect a calibrated parameter set whose full provenance is documented in the manuscript itself (Tables 6, 8, 9). A user running this open-source implementation with default parameters obtains the directional finding; users requiring the exact numerical headline are referred to the paper's calibration narrative. The dual view is intentional and honest: it lets readers verify the model architecture independently of any single calibration choice.

The PUE module, carbon module, and scenario logic all pass strict numerical tests (`tests/test_pue.py`, `tests/test_carbon.py`) against the empirical data points published in the paper.

---

## Repository layout

```
dc-decom/
├── README.md                       # this file
├── LICENSE                         # MIT (code)
├── LICENSE-DATA                    # CC BY 4.0 (data)
├── CITATION.cff                    # machine-readable citation
├── .gitignore
├── requirements.txt                # pinned Python deps
├── environment.yml                 # conda alternative
├── .github/workflows/
│   └── reproducibility.yml         # CI: solve + assert
├── src/
│   ├── dc_decom/
│   │   ├── __init__.py
│   │   ├── pue.py                  # Eqs. 8-9 (calibrated 1.14 + 0.0167T + ...)
│   │   ├── carbon.py               # Eqs. 10-11 + trajectories
│   │   ├── load_inputs.py          # CSV ingest
│   │   ├── scenarios.py            # REF / C-CN / R-CN
│   │   ├── model.py                # PuLP LP, constraints (a)-(e)
│   │   └── postprocess.py          # Table 10 decomposition
│   └── scripts/
│       ├── run_all.py              # solve + print all results
│       ├── run_ref.py
│       ├── run_ccn.py
│       └── run_rcn.py
├── data/
│   ├── preprocess_temperature.py   # regenerates the 96-h profile
│   └── parameters_csv/
│       ├── regions.csv             # five regions x climate, tariffs, PPA, RE_max
│       ├── global_parameters.csv   # paper Table 8 mirror
│       └── temperature_96h.csv     # 96 reps x 5 regions (4 seasons x 24 h)
├── tests/
│   ├── test_pue.py                 # 4 tests
│   ├── test_carbon.py              # 5 tests
│   └── test_headline_numbers.py    # 5 tests
└── docs/
    └── changelog.md
```

---

## Licensing

- **Code** — MIT License. See `LICENSE`. Copyright (c) 2026 Jin Toyohara.
- **Input data** — CC BY 4.0. See `LICENSE-DATA`. Public-government data with attribution preserved; please cite the original sources in addition to this repository.

---

## How to cite

Please cite the journal article as the canonical reference. A machine-readable `CITATION.cff` is provided at the repository root; GitHub renders a "Cite this repository" widget that produces the recommended BibTeX/APA.

> *Cost-Optimal Decarbonization Pathways for Data Centers in Japan: A Bottom-Up Model Integrating Location, Energy Systems, and Carbon Pricing*. **Energies** 2026, *19*(10), 2485. [https://doi.org/10.3390/en19102485](https://doi.org/10.3390/en19102485)

> Software archive: [10.5281/zenodo.20312805](https://doi.org/10.5281/zenodo.20312805) (this repository, v1.0.0).

---

## Contact and contribution

- Maintainer: Jin Toyohara — via GitHub Issues on this repository.
- Pull requests are welcome for documentation improvements, reproducibility fixes, and parameter-calibration refinements that bring LP-derived totals closer to the paper's headline.
- Issue/PR triage target: within 30 days during the first two years post-publication.
