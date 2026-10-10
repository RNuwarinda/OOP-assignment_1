import numpy as np
import pytest

from src.taxi import (
    ExponentialSmoothingForecaster,
    Forecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
)
from src.taxi_analysis import (
    compare_seasonal_naive,
    create_routes,
    evaluate_route_forecasters,
    forecast_day_11,
    generate_seasonal_demand,
    rolling_origin_predictions,
    run_analysis,
    solve_ntinda_equilibrium,
    tune_ses_alpha,
)


def test_route_revenue_and_statistics():
    route = Route("Example", [10, 20, 30], 2000)

    assert isinstance(route.passenger_counts, np.ndarray)
    assert route.daily_revenue().tolist() == [20000, 40000, 60000]
    assert route.total_revenue() == 120000
    stats = route.descriptive_statistics()
    assert stats["mean_passengers"] == pytest.approx(20)
    assert stats["median_passengers"] == pytest.approx(20)
    assert stats["variance_passengers"] == pytest.approx(100)
    assert stats["standard_deviation_passengers"] == pytest.approx(10)


@pytest.mark.parametrize(
    ("counts", "fare"),
    [([], 100), ([1, -1], 100), ([1, np.nan], 100), ([1, 2], -1), ([1], np.inf)],
)
def test_route_rejects_invalid_values(counts, fare):
    with pytest.raises(ValueError):
        Route("Invalid", counts, fare)


def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()


def test_moving_average_forecasts_recursively_for_multiple_steps():
    model = MovingAverageForecaster(window=3).fit([10, 20, 30])

    assert model.predict(3) == pytest.approx([20, 23.3333333333, 24.4444444444])


def test_exponential_smoothing_forecast_and_alpha_validation():
    model = ExponentialSmoothingForecaster(alpha=0.5).fit([10, 20, 30])

    assert model.level == pytest.approx(22.5)
    assert model.predict(2) == pytest.approx([22.5, 22.5])
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSmoothingForecaster(alpha=0)


def test_linear_trend_and_seasonal_naive_forecast():
    trend = LinearTrendForecaster().fit([10, 20, 30])
    seasonal = SeasonalNaiveForecaster(period=3).fit([10, 20, 30, 40, 50, 60])

    assert trend.predict(2) == pytest.approx([40, 50])
    assert seasonal.predict(3) == pytest.approx([40, 50, 60])


@pytest.mark.parametrize("horizon", [0, -2, 1.5, "2"])
def test_forecasters_reject_invalid_horizon(horizon):
    model = MovingAverageForecaster().fit([10, 20, 30])

    with pytest.raises(ValueError, match="positive integer"):
        model.predict(horizon)


def test_ntinda_equilibrium_and_current_fare_shortage():
    equilibrium = solve_ntinda_equilibrium()

    assert equilibrium["equilibrium_fare_ugx"] == pytest.approx(2200)
    assert equilibrium["equilibrium_quantity_passengers_per_trip_hour"] == pytest.approx(76)
    assert equilibrium["current_fare_position"] == "below"
    assert equilibrium["demand_at_current_fare"] == pytest.approx(80)
    assert equilibrium["supply_at_current_fare"] == pytest.approx(70)
    assert equilibrium["shortage_at_current_fare"] == pytest.approx(10)


def test_rolling_origin_uses_days_four_through_ten():
    values = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]
    result = rolling_origin_predictions(values, MovingAverageForecaster)

    assert result["origins"].tolist() == list(range(4, 11))
    assert result["actual"].tolist() == values[3:]
    assert len(result["predicted"]) == 7
    assert result["mae"] == pytest.approx(
        np.mean(np.abs(result["actual"] - result["predicted"]))
    )


def test_ses_grid_search_and_route_evaluation_choose_minimum_mae():
    route = create_routes()[0]
    alpha, scores = tune_ses_alpha(route.passenger_counts)
    evaluation = evaluate_route_forecasters(route)

    assert alpha in scores
    assert evaluation["ses_alpha"] == alpha
    assert scores[alpha] == pytest.approx(min(scores.values()))
    assert evaluation["selected_model"] in evaluation["model_results"]


def test_fleet_plan_uses_eight_trips_fourteen_seats_and_buffer():
    forecast = forecast_day_11(
        create_routes()[0], ExponentialSmoothingForecaster, alpha=0.9
    )

    assert forecast["capacity_per_vehicle_passengers"] == 112
    assert forecast["buffered_passengers"] == pytest.approx(
        forecast["forecast_passengers"] * 1.15
    )
    assert forecast["vehicles"] == int(
        np.ceil(forecast["forecast_passengers"] * 1.15 / 112)
    )
    assert forecast["forecast_revenue_ugx"] == pytest.approx(
        forecast["forecast_passengers"] * 2000
    )


def test_seasonal_demand_is_seeded_and_seasonal_naive_wins():
    demand_a = generate_seasonal_demand(days=60, seed=17)
    demand_b = generate_seasonal_demand(days=60, seed=17)
    for route in demand_a:
        np.testing.assert_array_equal(demand_a[route], demand_b[route])

    comparison = compare_seasonal_naive(days=60, seed=17)
    assert comparison["seasonal_naive_beats_moving_average"]
    assert (
        comparison["mean_mae_by_model"]["7-day seasonal naive"]
        < comparison["mean_mae_by_model"]["3-day moving average"]
    )


def test_full_analysis_produces_forecasts_tables_and_plots():
    result = run_analysis()

    assert len(result["routes"]) == 3
    assert result["model_mae"].shape == (9, 3)
    assert set(result["day_11_forecasts"]) == {
        "Kampala-Ntinda", "Kampala-Entebbe", "Kampala-Mukono"
    }
    assert result["revenue_plot_path"].is_file()
    assert result["seasonal_plot_path"].is_file()