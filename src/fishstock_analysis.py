from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .fishstock import FishStock, PriceModel, RiskAssessor
except ImportError:
    from fishstock import FishStock, PriceModel, RiskAssessor


HARVEST_SCENARIOS = (0.05, 0.10, 0.20, 0.30)
REFERENCE_HARVEST_RATE = 0.20
PRICE_START_UGX_PER_KG = 12000.0
TONNES_TO_KG = 1000.0


def fibonacci_baseline(count=15):
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    values = [1, 1]
    while len(values) < count:
        values.append(values[-1] + values[-2])
    return np.asarray(values[:count], dtype=np.int64)


def _revenue_paths(harvest_tonnes, prices_ugx_per_kg):
    harvest_kg = np.asarray(harvest_tonnes, dtype=float) * TONNES_TO_KG
    prices = np.asarray(prices_ugx_per_kg, dtype=float)
    if prices.ndim != 2 or prices.shape[1] != len(harvest_kg):
        raise ValueError("Price paths must have one price for each harvest week.")
    return np.sum(harvest_kg[np.newaxis, :] * prices, axis=1)


def _plot_results(scenarios, reference_revenues, reference_var_5, output_path):
    figure, (stock_axis, revenue_axis) = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = ("#2A9D8F", "#E9A23B", "#245B8F", "#BC4749")
    for (harvest_rate, scenario), color in zip(scenarios.items(), palette):
        weeks = np.arange(len(scenario["stock"]["stock_tonnes"]))
        stock_axis.plot(
            weeks,
            scenario["stock"]["stock_tonnes"],
            color=color,
            linewidth=2,
            label=f"h={harvest_rate:.2f}",
        )
    stock_axis.axhline(
        10000, color="#666666", linestyle=":", linewidth=1, label="Carrying capacity"
    )
    stock_axis.set_title("Weekly fish-stock trajectories")
    stock_axis.set_xlabel("Week")
    stock_axis.set_ylabel("Stock (tonnes)")
    stock_axis.grid(alpha=0.25)
    stock_axis.legend(frameon=False)

    revenue_axis.hist(
        reference_revenues / 1_000_000,
        bins=35,
        color="#31708E",
        alpha=0.82,
        edgecolor="white",
    )
    revenue_axis.axvline(
        reference_var_5 / 1_000_000,
        color="#B23A48",
        linewidth=2,
        linestyle="--",
        label=f"5th percentile: UGX {reference_var_5 / 1_000_000:,.1f}m",
    )
    revenue_axis.set_title(f"Annual revenue distribution (h={REFERENCE_HARVEST_RATE:.2f})")
    revenue_axis.set_xlabel("Annual revenue (million UGX)")
    revenue_axis.set_ylabel("Monte Carlo paths")
    revenue_axis.grid(axis="y", alpha=0.25)
    revenue_axis.legend(frameon=False)

    figure.suptitle("Lake Victoria Fish Stock and Export Revenue Risk", y=1.02)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_analysis(paths=1000, seed=2026, closed_season_weeks=8):
    if not isinstance(paths, int) or paths < 1000:
        raise ValueError("Use at least 1,000 Monte Carlo price paths.")
    stock_model = FishStock()
    risk_assessor = RiskAssessor(low_cv=0.10, moderate_cv=0.25)
    price_model = PriceModel(
        starting_price=PRICE_START_UGX_PER_KG,
        lower_bound=9000,
        upper_bound=16000,
        weekly_volatility=350,
        seed=seed,
    )
    price_paths = price_model.simulate_paths(weeks=52, paths=paths)
    scenarios = {}

    for harvest_rate in HARVEST_SCENARIOS:
        stock = stock_model.simulate(harvest_rate, weeks=52)
        revenues = _revenue_paths(stock["harvest_tonnes"], price_paths)
        revenue_risk = risk_assessor.assess_revenue(revenues)
        scenarios[harvest_rate] = {
            "stock": stock,
            "annual_revenues_UGX": revenues,
            "mean_total_revenue_UGX": revenue_risk["mean"],
            "risk": revenue_risk,
            "final_stock_tonnes": float(stock["stock_tonnes"][-1]),
            "total_harvest_tonnes": float(np.sum(stock["harvest_tonnes"])),
            "mean_harvest_tonnes_per_week": float(np.mean(stock["harvest_tonnes"])),
        }

    reference = scenarios[REFERENCE_HARVEST_RATE]
    five_year_weeks = 5 * 52
    five_year_prices = PriceModel(
        starting_price=PRICE_START_UGX_PER_KG,
        lower_bound=9000,
        upper_bound=16000,
        weekly_volatility=350,
        seed=seed + 1,
    ).simulate_paths(weeks=five_year_weeks, paths=paths)
    open_stock = stock_model.simulate(REFERENCE_HARVEST_RATE, weeks=five_year_weeks)
    closed_stock = stock_model.simulate(
        REFERENCE_HARVEST_RATE,
        weeks=five_year_weeks,
        closed_season_weeks=closed_season_weeks,
    )
    open_revenue = _revenue_paths(open_stock["harvest_tonnes"], five_year_prices)
    closed_revenue = _revenue_paths(closed_stock["harvest_tonnes"], five_year_prices)
    open_mean = float(np.mean(open_revenue))
    closed_mean = float(np.mean(closed_revenue))
    closed_season = {
        "harvest_rate": REFERENCE_HARVEST_RATE,
        "closed_weeks_per_year": closed_season_weeks,
        "open_final_stock_tonnes": float(open_stock["stock_tonnes"][-1]),
        "closed_final_stock_tonnes": float(closed_stock["stock_tonnes"][-1]),
        "open_mean_revenue_UGX": open_mean,
        "closed_mean_revenue_UGX": closed_mean,
        "mean_revenue_change_UGX": closed_mean - open_mean,
        "mean_revenue_change_percent": (
            (closed_mean - open_mean) / open_mean * 100 if open_mean else 0.0
        ),
        "open_revenue_paths_UGX": open_revenue,
        "closed_revenue_paths_UGX": closed_revenue,
    }

    plot_path = Path(__file__).resolve().parent.parent / "outputs" / "fish_stock_risk.png"
    _plot_results(
        scenarios,
        reference["annual_revenues_UGX"],
        reference["risk"]["var_5_revenue"],
        plot_path,
    )
    return {
        "fibonacci_baseline": fibonacci_baseline(15),
        "stock_model": stock_model,
        "price_paths": price_paths,
        "scenarios": scenarios,
        "reference_harvest_rate": REFERENCE_HARVEST_RATE,
        "maximum_sustainable_yield_tonnes_per_week": (
            stock_model.maximum_sustainable_yield
        ),
        "closed_season": closed_season,
        "plot_path": plot_path,
        "price_bounds_UGX_per_kg": (9000, 16000),
        "risk_cv_bands": {"Low": "CV < 10%", "Moderate": "10% <= CV < 25%", "High": "CV >= 25%"},
    }


def report_analysis(results):
    print("Lake Victoria Fish Stock & Export Risk Model")
    print(f"Fibonacci stock baseline (15 values): {results['fibonacci_baseline'].tolist()}")
    print(
        "Unbounded Fibonacci growth is biologically unrealistic: it has no carrying "
        "capacity, resource limitation, or density dependence. Real fish populations "
        "are constrained by habitat, food, recruitment, and mortality. Its accelerating "
        "sequence is a mathematical baseline, not a stock-dynamics model."
    )
    print(
        f"\nMSY = rK/4 = {results['maximum_sustainable_yield_tonnes_per_week']:,.0f} "
        "tonnes per week at the theoretical logistic equilibrium."
    )
    print("CV policy bands: Low <10%; Moderate 10-25%; High >=25% (illustrative).")
    print("\n52-week harvest scenarios, each with seeded Monte Carlo prices:")
    print(
        f"  {'h':>5} {'Final stock t':>15} {'Catch t':>12} {'Avg t/week':>12} "
        f"{'Mean revenue UGX':>19} {'5% revenue VaR':>19} {'CV':>8} {'Risk':>10}"
    )
    for harvest_rate, item in results["scenarios"].items():
        risk = item["risk"]
        print(
            f"  {harvest_rate:>5.2f} {item['final_stock_tonnes']:>15,.1f} "
            f"{item['total_harvest_tonnes']:>12,.1f} "
            f"{item['mean_harvest_tonnes_per_week']:>12,.1f} "
            f"{risk['mean']:>19,.0f} {risk['var_5_revenue']:>19,.0f} "
            f"{risk['coefficient_of_variation']:>7.1%} {risk['risk_class']:>10}"
        )

    reference_risk = results["scenarios"][results["reference_harvest_rate"]]["risk"]
    print("\nReference revenue statistics (h=0.20; UGX/year across price paths):")
    for key in ("mean", "median", "variance", "standard_deviation"):
        print(f"  {key.replace('_', ' ').title()}: {reference_risk[key]:,.2f}")
    print(f"  CV: {reference_risk['coefficient_of_variation']:.2%}")
    print(
        f"  5% revenue VaR (5th-percentile annual revenue): "
        f"UGX {reference_risk['var_5_revenue']:,.0f}"
    )
    print(
        f"  95% shortfall versus expected revenue: "
        f"UGX {reference_risk['loss_var_95_vs_mean']:,.0f}"
    )
    print(
        "  A raw variance cutoff of 50,000 is not portable: revenue variance has "
        "units UGX^2, so it changes quadratically with currency units and scales with "
        "catch volume and the time horizon. CV is dimensionless and normalized by mean."
    )

    closure = results["closed_season"]
    print(
        f"\nFive-year closed season ({closure['closed_weeks_per_year']} weeks/year, "
        f"h={closure['harvest_rate']:.2f}, same price paths):"
    )
    print(
        f"  Open season mean revenue: UGX {closure['open_mean_revenue_UGX']:,.0f}; "
        f"final stock {closure['open_final_stock_tonnes']:,.1f} t"
    )
    print(
        f"  Closed season mean revenue: UGX {closure['closed_mean_revenue_UGX']:,.0f}; "
        f"final stock {closure['closed_final_stock_tonnes']:,.1f} t"
    )
    print(
        f"  Revenue change: UGX {closure['mean_revenue_change_UGX']:,.0f} "
        f"({closure['mean_revenue_change_percent']:+.1f}%) over five years."
    )
    print(f"\nPlot saved to: {results['plot_path']}")


def main():
    results = run_analysis()
    report_analysis(results)


if __name__ == "__main__":
    main()