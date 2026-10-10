import numpy as np
import pytest

from src.ubos import (
    CAGRForecaster,
    DistrictPopulation,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
)


def test_district_population_stores_arrays_and_exposes_basic_interface():
    district = DistrictPopulation("Kampala", [2020, 2021], [1500, 1600])

    assert district.district_name == "Kampala"
    assert isinstance(district.years, np.ndarray)
    assert isinstance(district.population, np.ndarray)
    assert district.years.tolist() == [2020, 2021]
    assert district.population.tolist() == [1500.0, 1600.0]
    assert district.get_district_name() == "Kampala"
    np.testing.assert_array_equal(district.get_population(), [1500.0, 1600.0])
    assert len(district) == 2
    assert "Kampala" in repr(district)


@pytest.mark.parametrize(
    ("years", "population"),
    [
        ([], []),
        ([2020, 2021], [100]),
        ([2020, 2021], [100, -1]),
        ([2020, 2021], [100, np.nan]),
        ([2020, 2021], [100, np.inf]),
        ([[2020, 2021]], [100, 110]),
        ([2020, 2021], [[100, 110]]),
        ([2021, 2020], [100, 110]),
        ([2020, 2020], [100, 110]),
    ],
)
def test_district_population_rejects_invalid_input(years, population):
    with pytest.raises(ValueError):
        DistrictPopulation("Invalid", years, population)


def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()


def test_linear_trend_fit_and_predict():
    model = LinearTrendForecaster().fit(
        [2020, 2021, 2022], [100, 110, 120]
    )

    assert model.predict(2) == pytest.approx([130, 140])
    assert model.fitted_values == pytest.approx([100, 110, 120])


def test_cagr_fit_and_predict():
    model = CAGRForecaster().fit([2020, 2021, 2022], [100, 110, 121])

    assert model.annual_rate == pytest.approx(0.1)
    assert model.fitted_values == pytest.approx([100, 110, 121])
    assert model.predict(3) == pytest.approx([133.1, 146.41, 161.051])


def test_cagr_rejects_zero_starting_population():
    with pytest.raises(ValueError, match="positive first population"):
        CAGRForecaster().fit([2020, 2021], [0, 10])


def test_fibonacci_ratio_fit_and_predict():
    model = FibonacciRatioForecaster().fit([2020, 2021], [100, 150])

    assert model.fitted_values == pytest.approx([100, 150])
    assert model.predict(3) == pytest.approx([225, 375, 600])


@pytest.mark.parametrize(
    ("years", "population"),
    [
        ([], []),
        ([2020, 2021], [100]),
        ([2020, 2021], [100, -1]),
        ([2021, 2020], [100, 110]),
    ],
)
def test_fit_rejects_invalid_training_data(years, population):
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit(years, population)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "2"])
def test_predict_rejects_invalid_horizon(horizon):
    model = LinearTrendForecaster().fit([2020, 2021], [100, 110])

    with pytest.raises(ValueError, match="positive integer"):
        model.predict(horizon)