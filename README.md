OOP Assignment 1: UBOS District Population Forecaster
=====================================================

The OOP class definitions are in `src/ubos.py`; analysis code and the
reproducible Python entry point are in `src/ubos_analysis.py`. Run the analysis
from the project root:

    python src/ubos_analysis.py

For an interactive version, open and run `ubos_forecaster.ipynb` from top to
bottom. It imports the classes and analysis functions, displays the statistics,
validation and planning tables, and renders the saved plot.

The script reports descriptive statistics, annual growth rates, holdout model
metrics, selected models, 2025-2029 projections, and classroom estimates. It
saves the combined subplot figure, including 95% residual-bootstrap intervals,
to `outputs/district_forecast.png`. Kampala, Wakiso, and Gulu use the supplied
illustrative data. Mbarara and Mbale are additional illustrative data series.

Install the test dependencies and run the unit tests from the project root:

    python -m pip install -r requirements-dev.txt
    python -m pytest

The class tests are in `tests/test_ubos.py`.

Methods and interpretation
--------------------------

`statistics.variance` and `statistics.stdev` calculate sample statistics. By
contrast, `np.var` and `np.std` default to population statistics (`ddof=0`),
dividing squared deviations by N. For a sample estimate, use `ddof=1`, dividing
by N-1. This is why `np.var(values, ddof=1)` matches `statistics.variance`;
the corresponding sample standard deviation is `np.std(values, ddof=1)`.

The validation split uses 2015-2021 for fitting and 2022-2024 for testing. The
script reports MAE, RMSE, and MAPE for the linear trend, CAGR, and Fibonacci
ratio models, and selects the lowest-MAPE model for each district. Year-on-year
growth rates are percentage changes between adjacent observations. CAGR is the
constant annual relative growth rate that connects the first and last values.

The Fibonacci model multiplies the latest population by successive ratios
starting at 3/2, 5/3, 8/5, and so on. These ratios approach the golden ratio,
about 1.618, so this rule eventually implies roughly 61.8% growth every year.
It is not defensible as a demographic forecast without an unusual, independently
supported process that produces that extraordinary sustained growth. Its role
here is as a deliberately critical comparison, not a planning recommendation.

The reported variances compare the ten observed years with five forecast years.
They describe the spread of those values, not the uncertainty of a forecast;
their different time windows and trends also affect the comparison. Prediction
intervals instead bootstrap centered in-sample residuals and accumulate sampled
residual shocks across forecast years. With only ten observations, these
intervals are approximate and depend on residual behavior remaining relevant.

Classroom estimates assume 18% of the population is primary-school age and 53
pupils per classroom. The additional-room figure is 2029 required classrooms
minus 2024 required classrooms, with each year's classroom requirement rounded
up to a whole room. Population inputs and forecasts are expressed in thousands.

Kasese Micro-grid
-----------------

The micro-grid OOP solvers are in `src/microgrid.py`; input, generated data,
analysis, and plotting are in `src/microgrid_analysis.py`. Run the seeded
30-day CSV analysis (generating `data/kasese_demands.csv` if it does not exist):

    python src/microgrid_analysis.py

Use `python src/microgrid_analysis.py --csv path/to/demands.csv` to load a CSV,
or `python src/microgrid_analysis.py --interactive` to validate and solve one
day's demands interactively. The report saves `outputs/microgrid_usage_cost.png`.
The 30-day sum is labeled as the monthly cost for this illustrative month.
The import-based notebook walkthrough is `microgrid_analysis.ipynb`.

For the base matrix `[[3, 2], [4, 1]]`, the determinant is -5, so the equations
have a unique exact solution. Its 2-norm condition number is about 5.83: relative
input errors can be amplified by up to that factor in the worst perturbation
direction. Negative exact source usage is physically infeasible; the analysis
flags these days and uses non-negative least squares (NNLS), reporting the
remaining residual instead of silently clipping the solution.

Usage standard deviation measures absolute volatility, while coefficient of
variation measures volatility relative to each source's mean. Solar can have a
larger relative CV despite battery having the larger absolute standard deviation.
The sensitivity analysis uses 1,000 seeded demand perturbations of +/-5% and
compares the observed response with the matrix condition number.

`HybridMicroGrid` adds diesel and a total-supply equation. If its new row is a
linear combination of existing rows, the determinant is zero and the condition
number is infinite; the system cannot uniquely identify all three source values.
Micro-grid tests are in `tests/test_microgrid.py` and run with `python -m pytest`.

Lake Victoria Fish Stock
------------------------

The class-only stock, price, and risk models are in `src/fishstock.py`; the
scenario and Monte Carlo workflow is in `src/fishstock_analysis.py`. Run the
52-week harvest and price analysis with:

    python src/fishstock_analysis.py

Open `fish_stock_risk.ipynb` for the interactive results, scenario tables, and
plots. The analysis uses seeded bounded price paths, 1,000 simulations, and
provisional CV categories (Low <10%, Moderate 10-25%, High >=25%). Its 5% VaR
is the fifth percentile of annual revenue, not the loss percentile. Revenue
variance is measured in UGX squared, so a fixed raw threshold is not a
scale-independent risk rule. Harvest rates and the eight-week closed season are
illustrative model scenarios; validate them against local ecological and market
data before operational use. Fish-stock tests are in `tests/test_fishstock.py`.

Rainfall Pattern & Crop Suitability
-----------------------------------

The `Region` and `CropRule` classes are in `src/rainfall.py`; the classifier,
similarity/peak analyses, plotting, and NASA POWER download are in
`src/rainfall_analysis.py`. Run the illustrative three-region analysis with:

    python src/rainfall_analysis.py

Open `rainfall_crop_analyser.ipynb` for the crop/month classifications,
cosine/Pearson/Euclidean matrices, detected seasons, advisory note, and figures.
Crop rainfall cutoffs are explicitly screening proxies derived from FAO crop
water totals and season lengths, not locally calibrated agronomic prescriptions.
References: [FAO Irrigation Water Management Training Manual 3, Chapter 3](https://www.fao.org/4/s2022e/s2022e07.htm)
and the [World Bank Uganda historical climate portal](https://climateknowledgeportal.worldbank.org/country/uganda/climate-data-historical).

The optional 10-year extension uses the NASA POWER monthly point API and saves
`data/nasa_power_rainfall_2015_2024.csv` plus monthly box plots at
`outputs/nasa_monthly_rainfall_boxplots.png`. Its monthly rainfall totals are
computed from the API's `mm/day` values using each month's actual number of days.
The illustrative line/heatmap figure is `outputs/rainfall_crop_suitability.png`.
Run all project tests with `python -m pytest`; rainfall tests are in
`tests/test_rainfall.py`.
