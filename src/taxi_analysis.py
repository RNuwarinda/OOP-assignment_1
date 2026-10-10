from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve

try:
    from .taxi import (
        ExponentialSmoothingForecaster,
        LinearTrendForecaster,
        MovingAverageForecaster,
        Route,
        SeasonalNaiveForecaster,
    )
except ImportError:
    from taxi import (
        ExponentialSmoothingForecaster,
        LinearTrendForecaster,
        MovingAverageForecaster,
        Route,
        SeasonalNaiveForecaster,
    )


VEHICLE_SEATS = 14
ONE_WAY_TRIPS_PER_DAY = 8
FLEET_BUFFER = 0.15
SES_ALPHA_GRID = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))


def create_routes():
    return [
        Route(
            "Kampala-Ntinda",
            [35, 40, 42, 50, 55, 60, 48, 52, 47, 45],
            2000,
        ),
        Route(
            "Kampala-Entebbe",
            [60, 58, 65, 70, 72, 80, 75, 68, 66, 64],
            5000,
        ),
        Route(
            "Kampala-Mukono",
            [45, 47, 50, 49, 55, 62, 58, 53, 51, 50],
            3000,
        ),
    ]


def solve_ntinda_equilibrium():
    # Qd = 120 - 0.02P and Qs = 10 + 0.03P, set Qd = Qs.
    coefficients = np.array([[1.0, 0.02], [1.0, -0.03]])
    right_hand_side = np.array([120.0, 10.0])
    quantity, fare = solve(coefficients, right_hand_side, assume_a="gen")
    current_fare = 2000.0
    current_demand = 120.0 - 0.02 * current_fare
    current_supply = 10.0 + 0.03 * current_fare
    return {
        "coefficients": coefficients,
        "right_hand_side": right_hand_side,
        "equilibrium_quantity_passengers_per_trip_hour": float(quantity),
        "equilibrium_fare_ugx": float(fare),
        "current_fare_ugx": current_fare,
        "current_fare_position": "below" if current_fare < fare else "above",
        "fare_difference_ugx": float(fare - current_fare),
        "demand_at_current_fare": current_demand,
        "supply_at_current_fare": current_supply,
        "shortage_at_current_fare": current_demand - current_supply,
    }


def _forecast_one(model_type, history, alpha=None):
    if model_type is ExponentialSmoothingForecaster:
        model = model_type(alpha=alpha).fit(history)
    else:
        model = model_type().fit(history)
    return float(model.predict(1)[0])


def rolling_origin_predictions(observations, model_type, first_test_index=3, alpha=None):
    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or first_test_index < 1 or first_test_index >= len(values):
        raise ValueError("Need one-dimensional data and a valid rolling-origin start index.")
    predictions = []
    actual = []
    origins = []
    for test_index in range(first_test_index, len(values)):
        history = values[:test_index]
        predictions.append(_forecast_one(model_type, history, alpha))
        actual.append(float(values[test_index]))
        origins.append(test_index + 1)
    return {
        "origins": np.asarray(origins),
        "actual": np.asarray(actual),
        "predicted": np.asarray(predictions),
        "mae": float(np.mean(np.abs(np.asarray(actual) - np.asarray(predictions)))),
    }


def tune_ses_alpha(observations, alpha_grid=SES_ALPHA_GRID, first_test_index=3):
    scores = {
        float(alpha): rolling_origin_predictions(
            observations,
            ExponentialSmoothingForecaster,
            first_test_index=first_test_index,
            alpha=float(alpha),
        )["mae"]
        for alpha in alpha_grid
    }
    best_alpha = min(scores, key=scores.get)
    return float(best_alpha), scores


def evaluate_route_forecasters(route, first_test_index=3):
    alpha, alpha_scores = tune_ses_alpha(
        route.passenger_counts, first_test_index=first_test_index
    )
    models = {
        "3-day moving average": (MovingAverageForecaster, None),
        f"Simple exponential smoothing (alpha={alpha:.1f})": (
            ExponentialSmoothingForecaster, alpha
        ),
        "Linear trend": (LinearTrendForecaster, None),
    }
    forecasts = {}
    for label, (model_type, model_alpha) in models.items():
        forecasts[label] = rolling_origin_predictions(
            route.passenger_counts,
            model_type,
            first_test_index=first_test_index,
            alpha=model_alpha,
        )
    selected_model = min(forecasts, key=lambda label: forecasts[label]["mae"])
    return {
        "model_results": forecasts,
        "ses_alpha": alpha,
        "ses_alpha_grid_mae": alpha_scores,
        "selected_model": selected_model,
        "selected_model_type": models[selected_model][0],
        "selected_model_alpha": models[selected_model][1],
    }


def forecast_day_11(route, selected_model_type, alpha=None):
    demand = _forecast_one(selected_model_type, route.passenger_counts, alpha)
    demand = max(0.0, demand)
    revenue = demand * route.fare_ugx
    vehicle_daily_capacity = VEHICLE_SEATS * ONE_WAY_TRIPS_PER_DAY
    vehicles = ceil(demand * (1 + FLEET_BUFFER) / vehicle_daily_capacity)
    return {
        "forecast_passengers": demand,
        "fare_ugx": route.fare_ugx,
        "forecast_revenue_ugx": revenue,
        "capacity_per_vehicle_passengers": vehicle_daily_capacity,
        "buffered_passengers": demand * (1 + FLEET_BUFFER),
        "vehicles": vehicles,
    }


def _plot_revenue_forecasts(routes, evaluations, day_11_forecasts, output_path):
    figure, axis = plt.subplots(figsize=(12, 6.5))
    colors = ("#245B8F", "#C75B39", "#2A9D8F")
    days = np.arange(1, len(routes[0]) + 1)
    for route, color in zip(routes, colors):
        actual_revenue = route.daily_revenue()
        forecast = day_11_forecasts[route.name]["forecast_revenue_ugx"]
        axis.plot(days, actual_revenue / 1000, marker="o", linewidth=1.8,
                  color=color, label=f"{route.name} actual")
        axis.scatter([11], [forecast / 1000], marker="D", s=70, color=color,
                     edgecolor="black", linewidth=0.5,
                     label=f"{route.name} day 11 forecast")
        axis.plot([10, 11], [actual_revenue[-1] / 1000, forecast / 1000],
                  linestyle="--", linewidth=1, color=color, alpha=0.75)
    axis.axvline(10.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.set_title("Daily route revenue and selected day-11 forecast")
    axis.set_xlabel("Day")
    axis.set_ylabel("Revenue (thousand UGX)")
    axis.set_xticks(np.arange(1, 12))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_seasonal_demand(days=60, seed=2025):
    if not isinstance(days, int) or days < 14:
        raise ValueError("days must be an integer of at least 14 for seasonal evaluation.")
    rng = np.random.default_rng(seed)
    weekday_multipliers = np.array([0.92, 0.90, 0.93, 1.00, 1.30, 1.05, 0.68])
    route_bases = {
        "Kampala-Ntinda": 48.0,
        "Kampala-Entebbe": 68.0,
        "Kampala-Mukono": 52.0,
    }
    simulated = {}
    for route_name, baseline in route_bases.items():
        noise = rng.normal(0, 2.0, size=days)
        passengers = np.rint(
            baseline * weekday_multipliers[np.arange(days) % 7] + noise
        )
        simulated[route_name] = np.maximum(0, passengers).astype(int)
    return simulated


def compare_seasonal_naive(days=60, seed=2025, first_test_index=7):
    simulated = generate_seasonal_demand(days=days, seed=seed)
    rows = []
    detail = {}
    for route_name, passengers in simulated.items():
        moving = rolling_origin_predictions(
            passengers, MovingAverageForecaster, first_test_index=first_test_index
        )
        seasonal = rolling_origin_predictions(
            passengers, SeasonalNaiveForecaster, first_test_index=first_test_index
        )
        detail[route_name] = {"moving_average": moving, "seasonal_naive": seasonal}
        rows.extend([
            {"route": route_name, "model": "3-day moving average", "MAE": moving["mae"]},
            {"route": route_name, "model": "7-day seasonal naive", "MAE": seasonal["mae"]},
        ])
    comparison = pd.DataFrame(rows)
    average_scores = comparison.groupby("model")["MAE"].mean()
    return {
        "passengers": simulated,
        "comparison": comparison,
        "detail": detail,
        "mean_mae_by_model": average_scores.to_dict(),
        "seasonal_naive_beats_moving_average": (
            average_scores["7-day seasonal naive"] < average_scores["3-day moving average"]
        ),
    }


def plot_seasonal_extension(extension, output_path):
    figure, axis = plt.subplots(figsize=(12, 6))
    days = np.arange(1, len(next(iter(extension["passengers"].values()))) + 1)
    for route_name, values in extension["passengers"].items():
        axis.plot(days, values, linewidth=1.2, marker=".", label=route_name)
    axis.set_title("Synthetic 60-day passenger demand with weekly seasonality")
    axis.set_xlabel("Day")
    axis.set_ylabel("Passengers per day")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_analysis(seed=2025):
    routes = create_routes()
    equilibrium = solve_ntinda_equilibrium()
    route_evaluations = {}
    day_11_forecasts = {}
    model_mae_rows = []
    stats_rows = []

    for route in routes:
        evaluation = evaluate_route_forecasters(route, first_test_index=3)
        route_evaluations[route.name] = evaluation
        forecast = forecast_day_11(
            route,
            evaluation["selected_model_type"],
            evaluation["selected_model_alpha"],
        )
        day_11_forecasts[route.name] = {
            **forecast,
            "selected_model": evaluation["selected_model"],
        }
        for label, model_result in evaluation["model_results"].items():
            model_mae_rows.append({
                "route": route.name,
                "model": label,
                "MAE_passengers": model_result["mae"],
            })
        stats_rows.append({"route": route.name, **route.descriptive_statistics()})

    extension = compare_seasonal_naive(seed=seed)
    project_root = Path(__file__).resolve().parent.parent
    revenue_plot = _plot_revenue_forecasts(
        routes,
        route_evaluations,
        day_11_forecasts,
        project_root / "outputs" / "taxi_route_revenue_forecast.png",
    )
    seasonal_plot = plot_seasonal_extension(
        extension,
        project_root / "outputs" / "taxi_seasonal_demand.png",
    )
    return {
        "routes": routes,
        "equilibrium": equilibrium,
        "route_evaluations": route_evaluations,
        "model_mae": pd.DataFrame(model_mae_rows),
        "route_statistics": pd.DataFrame(stats_rows).set_index("route"),
        "day_11_forecasts": day_11_forecasts,
        "seasonal_extension": extension,
        "revenue_plot_path": revenue_plot,
        "seasonal_plot_path": seasonal_plot,
    }


def report_analysis(results):
    equilibrium = results["equilibrium"]
    print("Kampala Taxi Route Revenue, Pricing & Fleet Planner")
    print(
        f"\nNtinda equilibrium: fare UGX {equilibrium['equilibrium_fare_ugx']:,.0f}, "
        f"quantity {equilibrium['equilibrium_quantity_passengers_per_trip_hour']:.1f} "
        "passengers/trip-hour."
    )
    print(
        f"Current UGX {equilibrium['current_fare_ugx']:,.0f} fare is "
        f"{equilibrium['current_fare_position']} equilibrium by "
        f"UGX {equilibrium['fare_difference_ugx']:,.0f}."
    )
    print(
        f"At current fare, modeled demand is {equilibrium['demand_at_current_fare']:.0f} "
        f"and supply is {equilibrium['supply_at_current_fare']:.0f} passengers/trip-hour "
        f"(shortage {equilibrium['shortage_at_current_fare']:.0f})."
    )
    print("\nRoute passenger/revenue statistics:")
    print(results["route_statistics"].round(2).to_string())
    print("\nRolling-origin MAE, days 4-10 (passengers):")
    print(results["model_mae"].pivot(index="route", columns="model", values="MAE_passengers").round(2).to_string())
    print("\nDay-11 forecast and fleet (8 trips x 14 seats; 15% buffer):")
    fleet_rows = []
    for route, item in results["day_11_forecasts"].items():
        fleet_rows.append({
            "route": route,
            "selected_model": item["selected_model"],
            "forecast_passengers": item["forecast_passengers"],
            "forecast_revenue_ugx": item["forecast_revenue_ugx"],
            "vehicles": item["vehicles"],
        })
    print(pd.DataFrame(fleet_rows).set_index("route").round(2).to_string())
    print("\nSynthetic 60-day weekly-seasonality MAE:")
    print(results["seasonal_extension"]["comparison"].pivot(
        index="route", columns="model", values="MAE"
    ).round(2).to_string())
    print(
        "Seasonal naive beats moving average on mean MAE:",
        results["seasonal_extension"]["seasonal_naive_beats_moving_average"],
    )
    print(f"\nRevenue plot: {results['revenue_plot_path']}")
    print(f"Seasonal extension plot: {results['seasonal_plot_path']}")


def main():
    report_analysis(run_analysis())


if __name__ == "__main__":
    main()