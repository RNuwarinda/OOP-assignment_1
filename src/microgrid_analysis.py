import argparse
import csv
from pathlib import Path
import statistics
from timeit import repeat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve

try:
    from .microgrid import HybridMicroGrid, MicroGrid
except ImportError:
    from microgrid import HybridMicroGrid, MicroGrid


SOLAR_RATE_UGX = 150.0
BATTERY_RATE_UGX = 450.0
CSV_COLUMNS = ("day", "D1_kWh", "D2_kWh")


def read_nonnegative_number(prompt, input_fn=None, output_fn=None):
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn
    while True:
        raw_value = input_fn(prompt).strip()
        if not raw_value:
            output_fn("Value cannot be empty. Please enter a non-negative number.")
            continue
        try:
            value = float(raw_value)
        except ValueError:
            output_fn("Value must be numeric. Please try again.")
            continue
        if not np.isfinite(value) or value < 0:
            output_fn("Value must be finite and non-negative. Please try again.")
            continue
        return value


def read_interactive_day(grid=None, input_fn=None, output_fn=None):
    grid = MicroGrid() if grid is None else grid
    d1 = read_nonnegative_number("Daytime load D1 (kWh): ", input_fn, output_fn)
    d2 = read_nonnegative_number("Critical-equipment load D2 (kWh): ", input_fn, output_fn)
    dispatch = grid.solve_day(d1, d2)
    return {
        "D1_kWh": d1,
        "D2_kWh": d2,
        "solar_kWh": float(dispatch[0]),
        "battery_kWh": float(dispatch[1]),
        "physically_feasible": grid.is_physically_feasible(dispatch),
    }


def run_interactive_mode(input_fn=None, output_fn=None):
    grid = MicroGrid()
    result = read_interactive_day(grid, input_fn, output_fn)
    result["exact_solution_feasible"] = result["physically_feasible"]
    dispatch = np.array([result["solar_kWh"], result["battery_kWh"]])
    if not result["physically_feasible"]:
        dispatch, residual = grid.solve_day_nonnegative(
            result["D1_kWh"], result["D2_kWh"]
        )
        result["solar_kWh"] = float(dispatch[0])
        result["battery_kWh"] = float(dispatch[1])
        result["nnls_residual_kWh"] = residual
        output = print if output_fn is None else output_fn
        output(
            "Exact solution requires negative source usage; switched to NNLS "
            f"(residual {residual:.3f} kWh)."
        )
    result["physically_feasible"] = grid.is_physically_feasible(dispatch)
    result["daily_cost_UGX"] = float(calculate_energy_cost(dispatch.reshape(1, 2))[0])
    output = print if output_fn is None else output_fn
    output(
        f"Dispatch: solar={result['solar_kWh']:.2f} kWh, "
        f"battery={result['battery_kWh']:.2f} kWh; "
        f"cost=UGX {result['daily_cost_UGX']:,.2f}."
    )
    return result


def generate_demand_csv(path, days=30, seed=42):
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    weekday = np.arange(days) % 7
    daytime_pattern = np.array([4, 7, 9, 8, 5, -4, -6], dtype=float)
    critical_pattern = np.array([1, 3, 4, 3, 1, -3, -4], dtype=float)
    d1 = 78 + daytime_pattern[weekday] + rng.normal(0, 2.2, size=days)
    d2 = 40 + critical_pattern[weekday] + rng.normal(0, 1.5, size=days)
    demands = np.column_stack((d1, d2))
    demands = np.maximum(demands, 0.0)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        for day, (daytime, critical) in enumerate(demands, start=1):
            writer.writerow((day, f"{daytime:.3f}", f"{critical:.3f}"))
    return path


def load_demand_csv(path):
    path = Path(path)
    frame = pd.read_csv(path)
    missing_columns = set(CSV_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Demand CSV must contain at least one day.")
    demands = frame[["D1_kWh", "D2_kWh"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(demands)) or np.any(demands < 0):
        raise ValueError("CSV demands must be finite and non-negative.")
    return frame.loc[:, CSV_COLUMNS].copy()


def solve_days_loop(grid, demands):
    demands = _validate_demand_matrix(demands)
    return np.vstack([grid.solve_day(d1, d2) for d1, d2 in demands])


def solve_days_vectorized(grid, demands):
    demands = _validate_demand_matrix(demands)
    if not grid.is_well_posed:
        raise ValueError("The system is singular or too ill-conditioned to solve reliably.")
    return solve(grid.coefficients, demands.T, assume_a="gen").T


def _validate_demand_matrix(demands):
    values = np.asarray(demands, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("Demands must be a non-empty array with shape (days, 2).")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Demand values must be finite and non-negative.")
    return values


def benchmark_solvers(grid, demands, number=10, repeats=5):
    demands = _validate_demand_matrix(demands)
    loop_times = repeat(lambda: solve_days_loop(grid, demands), number=number, repeat=repeats)
    vectorized_times = repeat(
        lambda: solve_days_vectorized(grid, demands), number=number, repeat=repeats
    )
    loop_median = statistics.median(loop_times) / number
    vectorized_median = statistics.median(vectorized_times) / number
    return {
        "loop_seconds_per_batch": loop_median,
        "vectorized_seconds_per_batch": vectorized_median,
        "speedup": loop_median / vectorized_median if vectorized_median else float("inf"),
        "repeats": repeats,
        "number_per_repeat": number,
    }


def solve_nonnegative_days(grid, demands):
    demands = _validate_demand_matrix(demands)
    raw = solve_days_vectorized(grid, demands)
    infeasible = np.any(raw < -1e-9, axis=1)
    handled = raw.copy()
    residual_norms = np.zeros(len(demands), dtype=float)
    for index in np.flatnonzero(infeasible):
        handled[index], residual_norms[index] = grid.solve_day_nonnegative(*demands[index])
    return raw, handled, infeasible, residual_norms


def usage_statistics(dispatches):
    dispatches = np.asarray(dispatches, dtype=float)
    if dispatches.ndim != 2 or dispatches.shape[1] != 2 or len(dispatches) < 2:
        raise ValueError("At least two solar/battery dispatch rows are required.")
    if not np.all(np.isfinite(dispatches)) or np.any(dispatches < 0):
        raise ValueError("Dispatches must be finite and non-negative.")

    report = {}
    for column, source in enumerate(("solar", "battery")):
        values = dispatches[:, column].tolist()
        report[source] = {
            "mean_kWh": statistics.mean(values),
            "variance_kWh2": statistics.variance(values),
            "stdev_kWh": statistics.stdev(values),
            "coefficient_of_variation": (
                statistics.stdev(values) / statistics.mean(values)
                if statistics.mean(values) > 0
                else float("inf")
            ),
        }
    report["more_volatile_by_stdev"] = max(
        ("solar", "battery"), key=lambda source: report[source]["stdev_kWh"]
    )
    report["more_volatile_by_cv"] = max(
        ("solar", "battery"),
        key=lambda source: report[source]["coefficient_of_variation"],
    )
    return report


def calculate_energy_cost(dispatches):
    dispatches = np.asarray(dispatches, dtype=float)
    if dispatches.ndim != 2 or dispatches.shape[1] != 2:
        raise ValueError("Dispatches must have shape (days, 2).")
    if not np.all(np.isfinite(dispatches)) or np.any(dispatches < 0):
        raise ValueError("Dispatches must be finite and non-negative.")
    return (
        dispatches[:, 0] * SOLAR_RATE_UGX
        + dispatches[:, 1] * BATTERY_RATE_UGX
    )


def sensitivity_analysis(grid, demands, draws=1000, seed=2025, perturbation=0.05):
    demands = _validate_demand_matrix(demands)
    if draws < 1000:
        raise ValueError("Use at least 1,000 draws for the requested Monte Carlo analysis.")
    if not 0 <= perturbation < 1:
        raise ValueError("perturbation must be in the range [0, 1).")

    baseline_demands = np.mean(demands, axis=0)
    baseline_dispatch = grid.solve_day(*baseline_demands)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(1 - perturbation, 1 + perturbation, size=(draws, 2))
    perturbed_demands = baseline_demands * factors
    perturbed_dispatch = solve_days_vectorized(grid, perturbed_demands)
    absolute_moves = np.abs(perturbed_dispatch - baseline_dispatch)
    relative_input_moves = np.linalg.norm(
        (perturbed_demands - baseline_demands) / np.linalg.norm(baseline_demands),
        axis=1,
    )
    relative_output_moves = np.linalg.norm(
        (perturbed_dispatch - baseline_dispatch) / np.linalg.norm(baseline_dispatch),
        axis=1,
    )
    response_ratios = relative_output_moves / relative_input_moves
    return {
        "draws": draws,
        "perturbation_fraction": perturbation,
        "baseline_demands_kWh": baseline_demands,
        "baseline_dispatch_kWh": baseline_dispatch,
        "mean_absolute_move_kWh": np.mean(absolute_moves, axis=0),
        "p95_absolute_move_kWh": np.percentile(absolute_moves, 95, axis=0),
        "mean_absolute_move_percent_of_baseline": (
            np.mean(absolute_moves, axis=0) / np.abs(baseline_dispatch) * 100
        ),
        "p95_relative_response_ratio": float(np.percentile(response_ratios, 95)),
    }


def plot_usage_and_cost(frame, dispatches, daily_costs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    day = frame["day"].to_numpy()
    figure, usage_axis = plt.subplots(figsize=(12, 6))
    usage_axis.bar(day, dispatches[:, 0], label="Solar", color="#E9A23B")
    usage_axis.bar(
        day, dispatches[:, 1], bottom=dispatches[:, 0],
        label="Battery", color="#31708E",
    )
    usage_axis.set_xlabel("Day")
    usage_axis.set_ylabel("Energy usage (kWh)")
    usage_axis.set_title("Kasese Micro-grid: daily source usage and energy cost")
    usage_axis.set_xticks(day)
    usage_axis.grid(axis="y", alpha=0.25)

    cost_axis = usage_axis.twinx()
    cost_axis.plot(day, daily_costs, color="#9B2C2C", marker="o", linewidth=1.5,
                   label="Daily cost")
    cost_axis.set_ylabel("Daily energy cost (UGX)")
    handles_1, labels_1 = usage_axis.get_legend_handles_labels()
    handles_2, labels_2 = cost_axis.get_legend_handles_labels()
    usage_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_microgrid_analysis(
    csv_path=None,
    seed=42,
    timing_number=10,
    timing_repeats=5,
    monte_carlo_draws=1000,
):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "kasese_demands.csv" if csv_path is None else Path(csv_path)
    if not csv_path.exists():
        generate_demand_csv(csv_path, days=30, seed=seed)
    frame = load_demand_csv(csv_path)
    demands = frame[["D1_kWh", "D2_kWh"]].to_numpy(dtype=float)
    grid = MicroGrid()

    loop_solutions = solve_days_loop(grid, demands)
    vectorized_solutions = solve_days_vectorized(grid, demands)
    np.testing.assert_allclose(loop_solutions, vectorized_solutions, rtol=1e-12, atol=1e-12)
    timing = benchmark_solvers(grid, demands, timing_number, timing_repeats)

    raw, dispatches, infeasible, residuals = solve_nonnegative_days(grid, demands)
    daily_costs = calculate_energy_cost(dispatches)
    stats_report = usage_statistics(dispatches)
    monthly_cost = float(np.sum(daily_costs))
    sensitivity = sensitivity_analysis(
        grid, demands, draws=monte_carlo_draws, seed=seed + 1
    )
    plot_path = plot_usage_and_cost(
        frame, dispatches, daily_costs,
        project_root / "outputs" / "microgrid_usage_cost.png",
    )

    results = frame.copy()
    results["raw_solar_kWh"] = raw[:, 0]
    results["raw_battery_kWh"] = raw[:, 1]
    results["physically_infeasible"] = infeasible
    results["solar_kWh"] = dispatches[:, 0]
    results["battery_kWh"] = dispatches[:, 1]
    results["nnls_residual_kWh"] = residuals
    results["daily_cost_UGX"] = daily_costs

    return {
        "grid": grid,
        "condition_number": grid.condition_number,
        "determinant": grid.determinant,
        "data": results,
        "raw_solutions": raw,
        "dispatches": dispatches,
        "infeasible_days": results.loc[infeasible].copy(),
        "statistics": stats_report,
        "timing": timing,
        "monthly_cost_UGX": monthly_cost,
        "sensitivity": sensitivity,
        "plot_path": plot_path,
    }


def report_analysis(results):
    print("Kasese rural health centre micro-grid")
    print(
        f"Determinant: {results['determinant']:.3f}; "
        f"condition number (2-norm): {results['condition_number']:.3f}"
    )
    print(
        "A nonzero determinant means the two constraints have one unique exact solution. "
        "The moderate condition number means relative input errors are amplified by at "
        "most about 5.83 in the worst direction, not that every day's error grows that much."
    )
    print(f"\nLoaded {len(results['data'])} days; flagged infeasible days:")
    if results["infeasible_days"].empty:
        print("  None")
    else:
        print(
            results["infeasible_days"][
                ["day", "D1_kWh", "D2_kWh", "raw_solar_kWh", "raw_battery_kWh",
                 "solar_kWh", "battery_kWh", "nnls_residual_kWh"]
            ].to_string(index=False)
        )

    timing = results["timing"]
    print("\nMedian time per 30-day batch (timeit):")
    print(f"  Python loop: {timing['loop_seconds_per_batch'] * 1000:.3f} ms")
    print(f"  Vectorized solve: {timing['vectorized_seconds_per_batch'] * 1000:.3f} ms")
    print(
        "  The vectorized path sends a 2x30 right-hand side to one LAPACK solve, "
        "avoiding repeated Python calls. For only 30 days absolute times are tiny "
        "and timings depend on machine/load."
    )

    print("\nUsage statistics (sample variance and standard deviation):")
    for source in ("solar", "battery"):
        item = results["statistics"][source]
        print(
            f"  {source.title()}: mean={item['mean_kWh']:.2f} kWh, "
            f"variance={item['variance_kWh2']:.2f} kWh^2, "
            f"std={item['stdev_kWh']:.2f} kWh, "
            f"CV={item['coefficient_of_variation']:.1%}"
        )
    print(
        f"  More volatile by absolute standard deviation: "
        f"{results['statistics']['more_volatile_by_stdev']}; "
        f"by relative coefficient of variation: "
        f"{results['statistics']['more_volatile_by_cv']}."
    )
    print(f"\nTotal 30-day energy cost: UGX {results['monthly_cost_UGX']:,.0f}")
    print("First five daily costs (UGX):")
    print(results["data"][["day", "daily_cost_UGX"]].head().to_string(index=False))

    sensitivity = results["sensitivity"]
    print(
        f"\nSensitivity ({sensitivity['draws']:,} draws, independent demands ±"
        f"{sensitivity['perturbation_fraction']:.0%}):"
    )
    for name, index in (("solar", 0), ("battery", 1)):
        print(
            f"  {name.title()} mean absolute movement: "
            f"{sensitivity['mean_absolute_move_kWh'][index]:.3f} kWh "
            f"({sensitivity['mean_absolute_move_percent_of_baseline'][index]:.1f}% "
            "of baseline)"
        )
    print(
        f"  95th-percentile relative response ratio: "
        f"{sensitivity['p95_relative_response_ratio']:.3f}; "
        f"matrix condition number: {results['condition_number']:.3f}."
    )
    print(f"\nPlot saved to: {results['plot_path']}")


def main():
    parser = argparse.ArgumentParser(description="Kasese health-centre micro-grid analysis")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--interactive", action="store_true", help="enter one day's demands interactively"
    )
    input_group.add_argument(
        "--csv", type=Path, help="load daily D1_kWh and D2_kWh demands from a CSV file"
    )
    args = parser.parse_args()
    if args.interactive:
        run_interactive_mode()
        return
    results = run_microgrid_analysis(csv_path=args.csv)
    report_analysis(results)


if __name__ == "__main__":
    main()