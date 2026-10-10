import numpy as np
import pandas as pd
import pytest

from src.microgrid import HybridMicroGrid, MicroGrid
from src.microgrid_analysis import (
    calculate_energy_cost,
    generate_demand_csv,
    load_demand_csv,
    read_interactive_day,
    read_nonnegative_number,
    run_interactive_mode,
    sensitivity_analysis,
    solve_days_loop,
    solve_days_vectorized,
    solve_nonnegative_days,
    usage_statistics,
)


def test_microgrid_well_posed_diagnostics_and_solution():
    grid = MicroGrid()

    assert np.isclose(grid.determinant, -5)
    assert np.isclose(grid.condition_number, 5.82842712474619)
    assert grid.is_well_posed
    assert grid.solve_day(80, 45) == pytest.approx([2, 37])


@pytest.mark.parametrize(
    ("d1", "d2"),
    [(-1, 10), (10, -1), (np.nan, 10), (10, np.inf), ("invalid", 10)],
)
def test_microgrid_rejects_invalid_demands(d1, d2):
    with pytest.raises(ValueError):
        MicroGrid().solve_day(d1, d2)


def test_microgrid_rejects_invalid_or_ill_posed_matrices():
    with pytest.raises(ValueError, match="shape"):
        MicroGrid([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="finite"):
        MicroGrid([[1, np.nan], [4, 5]])

    dependent = MicroGrid([[3, 2], [6, 4]])
    assert not dependent.is_well_posed
    with pytest.raises(ValueError, match="singular or too ill-conditioned"):
        dependent.solve_day(10, 20)


def test_infeasible_exact_solution_is_flagged_and_nnls_is_nonnegative():
    grid = MicroGrid()
    raw = grid.solve_day(100, 40)
    handled, residual = grid.solve_day_nonnegative(100, 40)

    assert raw[0] < 0
    assert not grid.is_physically_feasible(raw)
    assert grid.is_physically_feasible(handled)
    assert handled == pytest.approx([0, 48])
    assert residual > 0


def test_interactive_input_reprompts_for_empty_non_numeric_and_negative():
    answers = iter(["", "abc", "-4", "75.5"])
    messages = []

    value = read_nonnegative_number(
        "Demand: ", input_fn=lambda _prompt: next(answers), output_fn=messages.append
    )

    assert value == 75.5
    assert len(messages) == 3


def test_interactive_day_reads_both_demands_and_solves():
    answers = iter(["80", "45"])
    result = read_interactive_day(
        input_fn=lambda _prompt: next(answers), output_fn=lambda _message: None
    )

    assert result["solar_kWh"] == pytest.approx(2)
    assert result["battery_kWh"] == pytest.approx(37)
    assert result["physically_feasible"]


def test_interactive_mode_reports_nnls_handling_for_infeasible_exact_solution():
    answers = iter(["100", "40"])
    messages = []
    result = run_interactive_mode(
        input_fn=lambda _prompt: next(answers), output_fn=messages.append
    )

    assert not result["exact_solution_feasible"]
    assert result["physically_feasible"]
    assert result["solar_kWh"] == pytest.approx(0)
    assert result["nnls_residual_kWh"] > 0
    assert result["daily_cost_UGX"] > 0
    assert any("switched to NNLS" in message for message in messages)


def test_seeded_csv_generation_and_loading(tmp_path):
    path_a = generate_demand_csv(tmp_path / "first.csv", days=30, seed=17)
    path_b = generate_demand_csv(tmp_path / "second.csv", days=30, seed=17)
    frame_a = load_demand_csv(path_a)
    frame_b = load_demand_csv(path_b)

    assert len(frame_a) == 30
    assert list(frame_a.columns) == ["day", "D1_kWh", "D2_kWh"]
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert frame_a["D1_kWh"].std() > 0
    assert frame_a["D2_kWh"].std() > 0


def test_csv_loader_rejects_missing_columns_and_negative_demands(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("day,D1_kWh\n1,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_demand_csv(missing)

    negative = tmp_path / "negative.csv"
    negative.write_text("day,D1_kWh,D2_kWh\n1,10,-2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite and non-negative"):
        load_demand_csv(negative)


def test_loop_and_vectorized_solves_match():
    demands = np.array([[80, 45], [100, 40], [70, 35]], dtype=float)
    grid = MicroGrid()

    np.testing.assert_allclose(
        solve_days_loop(grid, demands), solve_days_vectorized(grid, demands)
    )


def test_batch_nonnegative_strategy_flags_raw_infeasible_days():
    grid = MicroGrid()
    demands = np.array([[80, 45], [100, 40]], dtype=float)

    raw, handled, infeasible, residuals = solve_nonnegative_days(grid, demands)

    assert infeasible.tolist() == [False, True]
    assert raw[1, 0] < 0
    assert np.all(handled >= 0)
    assert residuals[0] == 0
    assert residuals[1] > 0


def test_usage_statistics_and_energy_cost():
    dispatches = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)
    stats = usage_statistics(dispatches)

    assert stats["solar"]["mean_kWh"] == pytest.approx(3)
    assert stats["solar"]["variance_kWh2"] == pytest.approx(4)
    assert stats["solar"]["stdev_kWh"] == pytest.approx(2)
    assert stats["more_volatile_by_stdev"] == "battery"
    assert calculate_energy_cost(dispatches) == pytest.approx([4650, 9450, 14250])


def test_sensitivity_is_seeded_and_uses_one_thousand_draws():
    grid = MicroGrid()
    demands = np.tile([80.0, 45.0], (30, 1))
    result_a = sensitivity_analysis(grid, demands, draws=1000, seed=12)
    result_b = sensitivity_analysis(grid, demands, draws=1000, seed=12)

    np.testing.assert_array_equal(
        result_a["mean_absolute_move_kWh"], result_b["mean_absolute_move_kWh"]
    )
    assert result_a["draws"] == 1000
    assert np.all(result_a["mean_absolute_move_kWh"] > 0)
    assert result_a["p95_relative_response_ratio"] < grid.condition_number


def test_hybrid_grid_solves_three_constraints():
    grid = HybridMicroGrid()

    assert np.isclose(grid.determinant, -4)
    assert grid.is_well_posed
    dispatch = grid.solve_day(80, 45, 50)
    np.testing.assert_allclose(grid.coefficients @ dispatch, [80, 45, 50])
    assert len(dispatch) == 3


def test_hybrid_grid_detects_linearly_dependent_third_equation():
    dependent = HybridMicroGrid(
        [
            [3, 2, 1],
            [4, 1, 2],
            [7, 3, 3],
        ]
    )

    assert not dependent.is_well_posed
    with pytest.raises(ValueError, match="singular or too ill-conditioned"):
        dependent.solve_day(80, 45, 125)