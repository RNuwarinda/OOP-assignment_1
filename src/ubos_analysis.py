from math import ceil
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

try:
    from .ubos import (
        CAGRForecaster,
        DistrictPopulation,
        FibonacciRatioForecaster,
        LinearTrendForecaster,
    )
except ImportError:
    from ubos import (
        CAGRForecaster,
        DistrictPopulation,
        FibonacciRatioForecaster,
        LinearTrendForecaster,
    )


MODEL_TYPES = (LinearTrendForecaster, CAGRForecaster, FibonacciRatioForecaster)


def create_districts():
    years = np.arange(2015, 2025)
    return [
        DistrictPopulation(
            "Kampala", years,
            [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
        ),
        DistrictPopulation(
            "Wakiso", years,
            [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
        ),
        DistrictPopulation(
            "Gulu", years,
            [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
        ),
        DistrictPopulation(
            "Mbarara", years,
            [480, 495, 510, 525, 545, 565, 585, 605, 630, 655],
        ),
        DistrictPopulation(
            "Mbale", years,
            [520, 535, 550, 570, 590, 610, 635, 660, 690, 720],
        ),
    ]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    errors = actual - np.asarray(predicted, dtype=float)
    nonzero_actual = actual != 0
    mape = (
        np.mean(np.abs(errors[nonzero_actual] / actual[nonzero_actual])) * 100
        if np.any(nonzero_actual)
        else float("nan")
    )
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": float(mape),
    }


def bootstrap_interval(residuals, forecast, resamples=1000, rng=None):
    if resamples < 1000:
        raise ValueError("Use at least 1,000 resamples for the requested interval.")
    rng = np.random.default_rng() if rng is None else rng
    residuals = np.asarray(residuals, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    centered_residuals = residuals - np.mean(residuals)
    sampled_shocks = rng.choice(
        centered_residuals, size=(resamples, len(forecast)), replace=True
    )
    forecast_paths = forecast[np.newaxis, :] + np.cumsum(sampled_shocks, axis=1)
    lower = np.maximum(0, np.percentile(forecast_paths, 2.5, axis=0))
    upper = np.percentile(forecast_paths, 97.5, axis=0)
    return lower, upper


def descriptive_statistics(district):
    values = district.population
    as_list = values.tolist()
    return {
        "statistics_mean": statistics.mean(as_list),
        "statistics_median": statistics.median(as_list),
        "statistics_variance": statistics.variance(as_list),
        "statistics_stdev": statistics.stdev(as_list),
        "numpy_mean": float(np.mean(values)),
        "numpy_median": float(np.median(values)),
        "numpy_variance_ddof0": float(np.var(values)),
        "numpy_stdev_ddof0": float(np.std(values)),
        "numpy_variance_ddof1": float(np.var(values, ddof=1)),
        "numpy_stdev_ddof1": float(np.std(values, ddof=1)),
    }


def _plot_results(plot_data, output_path):
    figure, axes = plt.subplots(
        len(plot_data), 1, figsize=(11, 3.6 * len(plot_data)), sharex=True
    )
    for axis, item in zip(axes, plot_data):
        district = item["district"]
        axis.plot(
            district.years, district.population, "o-", color="#22577A",
            label="Actual", linewidth=1.8,
        )
        axis.plot(
            item["train_years"], item["fitted"], "s--", color="#57A773",
            label="Fitted (selected model)", linewidth=1.5,
        )
        axis.plot(
            item["test_years"], item["test_prediction"], "D:", color="#D17A22",
            label="Holdout prediction", linewidth=1.7,
        )
        future_years = item["future_years"]
        forecast = item["forecast"]
        axis.plot(
            np.concatenate(([district.years[-1]], future_years)),
            np.concatenate(([district.population[-1]], forecast)),
            "o--", color="#B23A48", label="2025-2029 forecast", linewidth=1.8,
        )
        axis.fill_between(
            future_years, item["lower"], item["upper"], color="#B23A48",
            alpha=0.16, label="95% residual-bootstrap interval",
        )
        axis.axvline(2021.5, color="#555555", linestyle="--", linewidth=1)
        axis.text(
            2021.65, 0.04, "Train | test", transform=axis.get_xaxis_transform(),
            fontsize=8, color="#444444",
        )
        axis.set_title(f"{district.district_name} | {item['model_name']}", loc="left")
        axis.set_ylabel("Population (thousands)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)

    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(np.arange(2015, 2030, 2))
    figure.suptitle("UBOS District Population: actuals, validation, and forecasts", y=1.002)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_analysis(resamples=1000, random_seed=2025, report=True):
    districts = create_districts()
    train_mask = districts[0].years <= 2021
    test_mask = districts[0].years >= 2022
    rng = np.random.default_rng(random_seed)
    validation = {}
    choices = {}
    forecasts = {}
    plot_data = []

    if report:
        print("UBOS District Population Forecaster")
        print("Population values are in thousands; Mbarara and Mbale are illustrative additions.")

    for district in districts:
        stats = descriptive_statistics(district)
        annual_growth = np.diff(district.population) / district.population[:-1] * 100
        cagr = (district.population[-1] / district.population[0]) ** (
            1 / (district.years[-1] - district.years[0])
        ) - 1

        if report:
            print(f"\n{district.district_name}: descriptive statistics (population in thousands)")
            print(
                f"  statistics: mean={stats['statistics_mean']:.2f}, "
                f"median={stats['statistics_median']:.2f}, "
                f"sample variance={stats['statistics_variance']:.2f}, "
                f"sample std={stats['statistics_stdev']:.2f}"
            )
            print(
                f"  NumPy:      mean={stats['numpy_mean']:.2f}, "
                f"median={stats['numpy_median']:.2f}, "
                f"population variance={stats['numpy_variance_ddof0']:.2f}, "
                f"population std={stats['numpy_stdev_ddof0']:.2f}, "
                f"sample variance (ddof=1)={stats['numpy_variance_ddof1']:.2f}"
            )
            print(
                "  Year-on-year growth (%): "
                f"{', '.join(f'{rate:.2f}' for rate in annual_growth)}"
            )
            print(f"  CAGR, 2015-2024: {cagr * 100:.2f}%")
            print(
                f"\n{district.district_name}: holdout validation "
                "(train through 2021; test 2022-2024)"
            )
            print(f"  {'Model':<27} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")

        district_validation = {}
        train_years = district.years[train_mask]
        train_population = district.population[train_mask]
        test_population = district.population[test_mask]
        for model_type in MODEL_TYPES:
            model = model_type().fit(train_years, train_population)
            scores = calculate_metrics(
                test_population, model.predict(len(test_population))
            )
            district_validation[model_type.__name__] = scores
            if report:
                print(
                    f"  {model_type.__name__:<27} {scores['MAE']:>10.2f} "
                    f"{scores['RMSE']:>10.2f} {scores['MAPE']:>9.2f}%"
                )

        selected_name = min(
            district_validation,
            key=lambda name: district_validation[name]["MAPE"],
        )
        selected_type = next(
            model_type for model_type in MODEL_TYPES
            if model_type.__name__ == selected_name
        )
        validation[district.district_name] = district_validation
        choices[district.district_name] = selected_type
        if report:
            print(f"  Selected by lowest holdout MAPE: {selected_name}")

    fastest = max(
        districts,
        key=lambda district: (district.population[-1] / district.population[0]) ** (
            1 / (district.years[-1] - district.years[0])
        ),
    )
    if report:
        print(f"\nFastest relative growth (CAGR): {fastest.district_name}")
        print("\nSelected-model forecasts and school planning")
        print(
            f"  {'District':<12} {'Model':<27} {'2029 pop. (000s)':>16} "
            f"{'2029 classrooms':>17} {'Additional':>12}"
        )

    for district in districts:
        model_type = choices[district.district_name]
        model = model_type().fit(district.years, district.population)
        future_years = np.arange(2025, 2030)
        forecast = model.predict(len(future_years))
        residuals = district.population - model.fitted_values
        lower, upper = bootstrap_interval(residuals, forecast, resamples, rng)
        current_classrooms = ceil(district.population[-1] * 1000 * 0.18 / 53)
        future_classrooms = ceil(forecast[-1] * 1000 * 0.18 / 53)
        additional = max(0, future_classrooms - current_classrooms)
        forecast_record = {
            "model": model_type.__name__,
            "years": future_years,
            "population": forecast,
            "lower_95": lower,
            "upper_95": upper,
            "actual_variance": float(np.var(district.population)),
            "forecast_variance": float(np.var(forecast)),
            "primary_age_population_2029": float(forecast[-1] * 1000 * 0.18),
            "classrooms_2024": current_classrooms,
            "classrooms_2029": future_classrooms,
            "additional_classrooms": additional,
        }
        forecasts[district.district_name] = forecast_record

        if report:
            print(
                f"  {district.district_name:<12} {model_type.__name__:<27} "
                f"{forecast[-1]:>16.2f} {future_classrooms:>17} {additional:>12}"
            )
            print(
                f"    Variance (thousands squared): actual 2015-2024="
                f"{forecast_record['actual_variance']:.2f}; "
                f"forecast 2025-2029={forecast_record['forecast_variance']:.2f}"
            )
            print(
                f"    2029 primary-age estimate: "
                f"{forecast_record['primary_age_population_2029']:,.0f} pupils; "
                f"95% bootstrap interval: {lower[-1]:.2f}-{upper[-1]:.2f} thousand"
            )

        validation_model = model_type().fit(train_years, train_population)
        plot_data.append({
            "district": district,
            "model_name": model_type.__name__,
            "train_years": train_years,
            "fitted": validation_model.fitted_values,
            "test_years": district.years[test_mask],
            "test_prediction": validation_model.predict(int(np.sum(test_mask))),
            "future_years": future_years,
            "forecast": forecast,
            "lower": lower,
            "upper": upper,
        })

    output_path = Path(__file__).resolve().parent.parent / "outputs" / "district_forecast.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_results(plot_data, output_path)
    if report:
        print(f"\nPlot saved to: {output_path}")
        print("Classroom estimates compare 2029 demand with 2024 demand, rounded up to whole rooms.")

    return {
        "districts": districts,
        "validation": validation,
        "selected_models": {name: model.__name__ for name, model in choices.items()},
        "fastest_relative_growth": fastest.district_name,
        "forecasts": forecasts,
        "plot_path": output_path,
    }


if __name__ == "__main__":
    run_analysis()