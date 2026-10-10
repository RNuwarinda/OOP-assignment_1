OOP Assignment 1: Applied Modelling Projects
============================================

**Name:** Nuwarinda Roseline  
**Student number:** S26M19/017  
**Access number:** B38638

## About

This repository contains five object-oriented Python mini-projects. Each keeps
its domain classes separate from its analysis workflow and includes a Jupyter
notebook for interactive results:

- District population forecasting and classroom planning
- Kasese health-centre micro-grid dispatch and cost analysis
- Lake Victoria fish-stock and export-revenue risk modelling
- Regional rainfall patterns and crop-suitability screening
- Kampala taxi-route forecasting and fleet planning

The examples use assignment-provided, illustrative, generated, or explicitly
identified NASA POWER data. They are educational models, not operational
forecasts or professional planning advice.

## Key Findings

- **District populations:** CAGR had the lowest 2022–2024 holdout MAPE for all
  five districts. Wakiso had the fastest historical relative growth (6.47%) and
  the largest estimated classroom increase (2,088) under the stated assumptions.
- **Micro-grid:** The two-source system has determinant -5 and condition number
  about 5.83. For the seeded 30-day example, 15 days required non-negative
  least-squares handling; modeled total energy cost was about UGX 543,741.
- **Fish stock:** Of the tested harvest rates, 0.20/week is closest to the
  logistic model's theoretical MSY point. An eight-week annual closure increased
  five-year ending stock but reduced modeled revenue by about 10.6%.
- **Rainfall:** The supplied profiles classify Kampala and Mbarara as bimodal
  and Gulu as unimodal. NASA POWER's 2015–2024 Gulu normal produced a different
  peak classification, illustrating sensitivity to data source and method.
- **Taxi routes:** SES with α=0.9 had the lowest short holdout MAE on all three
  routes. The seasonal-naïve model beat the moving average on deliberately
  seasonal synthetic data (mean MAE 2.65 vs 8.71 passengers).

These results depend on short series, simplified assumptions, and—in several
projects—synthetic inputs. Consult each notebook's Findings & Limitations and
References sections before interpreting or reusing a result.

## Setup and Tests

From the repository root, create an environment, install dependencies, and run
the test suite:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The test suite covers the OOP classes, analysis calculations, input validation,
and generated data workflows.

## Continuous Integration

The GitHub Actions workflow at `.github/workflows/tests.yml` runs on pushes,
pull requests, and manual dispatches. It sets up Python 3.12, installs
`requirements-dev.txt`, and runs `python -m pytest -q`.

## Run the Projects

Run each analysis script from the repository root:

| Project | Command | Notebook |
| --- | --- | --- |
| District population | `python src/ubos_analysis.py` | [ubos_forecaster.ipynb](ubos_forecaster.ipynb) |
| Micro-grid | `python src/microgrid_analysis.py` | [microgrid_analysis.ipynb](microgrid_analysis.ipynb) |
| Fish-stock risk | `python src/fishstock_analysis.py` | [fish_stock_risk.ipynb](fish_stock_risk.ipynb) |
| Rainfall and crops | `python src/rainfall_analysis.py` | [rainfall_crop_analyser.ipynb](rainfall_crop_analyser.ipynb) |
| Taxi fleet planner | `python src/taxi_analysis.py` | [taxi_fleet_planner.ipynb](taxi_fleet_planner.ipynb) |

The micro-grid script also accepts `--interactive` or `--csv path/to/file.csv`.
The rainfall NASA POWER extension is available from its notebook workflow.

Generated datasets and plots are saved under `data/` and `outputs/`.

## AI Usage

AI assistance was used to help develop and refine the pytest suite, GitHub
Actions workflow, this README, and project documentation. AI-generated
suggestions were reviewed and checked against the implementation and test
results.
