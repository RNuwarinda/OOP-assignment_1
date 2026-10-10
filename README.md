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
