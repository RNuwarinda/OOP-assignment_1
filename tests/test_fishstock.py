import numpy as np
import pytest

from src.fishstock import FishStock, PriceModel, RiskAssessor
from src.fishstock_analysis import (
    _revenue_paths,
    fibonacci_baseline,
    run_analysis,
)


def test_fibonacci_baseline_returns_requested_sequence():
    assert fibonacci_baseline(15).tolist() == [
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610
    ]


def test_logistic_harvest_simulation_reaches_expected_equilibrium():
    model = FishStock()
    simulation = model.simulate(harvest_rate=0.20, weeks=52)

    assert simulation["stock_tonnes"].shape == (53,)
    assert simulation["harvest_tonnes"].shape == (52,)
    assert simulation["stock_tonnes"][0] == pytest.approx(4000)
    assert simulation["stock_tonnes"][-1] == pytest.approx(5000, abs=0.02)
    assert model.equilibrium_stock(0.20) == pytest.approx(5000)
    assert model.maximum_sustainable_yield == pytest.approx(1000)


def test_closed_season_suspends_harvesting_for_last_eight_weeks_each_year():
    simulation = FishStock().simulate(
        harvest_rate=0.20, weeks=104, closed_season_weeks=8
    )

    assert np.flatnonzero(simulation["closed_weeks"]).tolist() == [
        *range(44, 52), *range(96, 104)
    ]
    assert np.all(simulation["harvest_tonnes"][44:52] == 0)
    assert np.all(simulation["harvest_tonnes"][96:104] == 0)
    assert simulation["stock_tonnes"][52] > simulation["stock_tonnes"][44]


@pytest.mark.parametrize(
    ("harvest_rate", "weeks", "closed_season_weeks"),
    [(-0.1, 52, 0), (1.1, 52, 0), (0.1, 0, 0), (0.1, 52, 53)],
)
def test_fish_stock_rejects_invalid_simulation_parameters(
    harvest_rate, weeks, closed_season_weeks
):
    with pytest.raises(ValueError):
        FishStock().simulate(harvest_rate, weeks, closed_season_weeks)


def test_price_paths_are_seeded_bounded_and_start_at_configured_price():
    first = PriceModel(seed=19).simulate_paths(weeks=52, paths=1000)
    second = PriceModel(seed=19).simulate_paths(weeks=52, paths=1000)

    assert first.shape == (1000, 52)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 0] == 12000)
    assert np.min(first) >= 9000
    assert np.max(first) <= 16000


def test_revenue_converts_harvest_tonnes_to_kg():
    revenue = _revenue_paths([1, 2], [[10000, 12000], [15000, 14000]])

    assert revenue == pytest.approx([34_000_000, 43_000_000])


def test_risk_assessor_uses_sample_statistics_and_cv_bands():
    assessor = RiskAssessor(low_cv=0.10, moderate_cv=0.25)
    assessed = assessor.assess_revenue([90, 100, 110])

    assert assessed["mean"] == pytest.approx(100)
    assert assessed["median"] == pytest.approx(100)
    assert assessed["variance"] == pytest.approx(100)
    assert assessed["standard_deviation"] == pytest.approx(10)
    assert assessed["coefficient_of_variation"] == pytest.approx(0.10)
    assert assessed["risk_class"] == "Moderate"
    assert assessed["var_5_revenue"] == pytest.approx(91)
    assert assessed["loss_var_95_vs_mean"] == pytest.approx(9)
    assert assessor.classify(0.09) == "Low"
    assert assessor.classify(0.25) == "High"


def test_analysis_runs_all_scenarios_and_closed_season_monte_carlo():
    result = run_analysis(paths=1000, seed=2026)

    assert result["fibonacci_baseline"].shape == (15,)
    assert set(result["scenarios"]) == {0.05, 0.10, 0.20, 0.30}
    assert all(
        scenario["annual_revenues_UGX"].shape == (1000,)
        for scenario in result["scenarios"].values()
    )
    assert result["scenarios"][0.20]["mean_harvest_tonnes_per_week"] == pytest.approx(
        result["maximum_sustainable_yield_tonnes_per_week"], rel=0.03
    )
    closure = result["closed_season"]
    assert closure["closed_weeks_per_year"] == 8
    assert closure["closed_final_stock_tonnes"] > closure["open_final_stock_tonnes"]
    assert closure["mean_revenue_change_UGX"] < 0
    assert result["plot_path"].is_file()


def test_analysis_rejects_fewer_than_one_thousand_paths():
    with pytest.raises(ValueError, match="at least 1,000"):
        run_analysis(paths=999)