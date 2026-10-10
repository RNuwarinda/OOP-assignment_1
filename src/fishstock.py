import statistics

import numpy as np


class FishStock:
    def __init__(self, initial_stock=4000.0, growth_rate=0.4, carrying_capacity=10000.0):
        self.initial_stock = float(initial_stock)
        self.growth_rate = float(growth_rate)
        self.carrying_capacity = float(carrying_capacity)
        if not np.all(np.isfinite([
            self.initial_stock, self.growth_rate, self.carrying_capacity
        ])):
            raise ValueError("Stock parameters must be finite.")
        if self.initial_stock < 0 or self.carrying_capacity <= 0:
            raise ValueError("Initial stock must be non-negative and capacity positive.")
        if self.initial_stock > self.carrying_capacity:
            raise ValueError("Initial stock cannot exceed carrying capacity.")
        if self.growth_rate < 0:
            raise ValueError("Growth rate must be non-negative.")

    def simulate(self, harvest_rate, weeks=52, closed_season_weeks=0):
        harvest_rate = float(harvest_rate)
        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1 per week.")
        if not isinstance(weeks, (int, np.integer)) or weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        if (
            not isinstance(closed_season_weeks, (int, np.integer))
            or not 0 <= closed_season_weeks <= 52
        ):
            raise ValueError("closed_season_weeks must be an integer from 0 to 52.")

        stocks = np.empty(weeks + 1, dtype=float)
        harvests = np.empty(weeks, dtype=float)
        closed = np.zeros(weeks, dtype=bool)
        stocks[0] = self.initial_stock
        closure_start = 52 - closed_season_weeks

        for week in range(weeks):
            is_closed = closed_season_weeks > 0 and week % 52 >= closure_start
            closed[week] = is_closed
            effective_harvest_rate = 0.0 if is_closed else harvest_rate
            current_stock = stocks[week]
            harvests[week] = effective_harvest_rate * current_stock
            stocks[week + 1] = (
                current_stock
                + self.growth_rate
                * current_stock
                * (1 - current_stock / self.carrying_capacity)
                - harvests[week]
            )

        return {
            "stock_tonnes": stocks,
            "harvest_tonnes": harvests,
            "closed_weeks": closed,
            "harvest_rate": harvest_rate,
        }

    @property
    def maximum_sustainable_yield(self):
        return self.growth_rate * self.carrying_capacity / 4

    def equilibrium_stock(self, harvest_rate):
        harvest_rate = float(harvest_rate)
        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1 per week.")
        if harvest_rate >= self.growth_rate:
            return 0.0
        return self.carrying_capacity * (1 - harvest_rate / self.growth_rate)


class PriceModel:
    def __init__(
        self,
        starting_price=12000.0,
        lower_bound=9000.0,
        upper_bound=16000.0,
        weekly_volatility=350.0,
        seed=42,
    ):
        self.starting_price = float(starting_price)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.weekly_volatility = float(weekly_volatility)
        self.seed = seed
        parameters = [
            self.starting_price,
            self.lower_bound,
            self.upper_bound,
            self.weekly_volatility,
        ]
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Price model parameters must be finite.")
        if self.lower_bound > self.starting_price or self.starting_price > self.upper_bound:
            raise ValueError("Starting price must be within the price bounds.")
        if self.lower_bound < 0 or self.weekly_volatility < 0:
            raise ValueError("Price bounds and volatility must be non-negative.")
        self._rng = np.random.default_rng(seed)

    def simulate_paths(self, weeks=52, paths=1, seed=None):
        if not isinstance(weeks, (int, np.integer)) or weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        if not isinstance(paths, (int, np.integer)) or paths < 1:
            raise ValueError("paths must be a positive integer.")
        rng = self._rng if seed is None else np.random.default_rng(seed)
        prices = np.empty((paths, weeks), dtype=float)
        prices[:, 0] = self.starting_price
        if weeks > 1:
            changes = rng.normal(0, self.weekly_volatility, size=(paths, weeks - 1))
            prices[:, 1:] = self.starting_price + np.cumsum(changes, axis=1)
            np.clip(prices, self.lower_bound, self.upper_bound, out=prices)
        return prices


class RiskAssessor:
    def __init__(self, low_cv=0.10, moderate_cv=0.25):
        self.low_cv = float(low_cv)
        self.moderate_cv = float(moderate_cv)
        if not np.isfinite(self.low_cv) or not np.isfinite(self.moderate_cv):
            raise ValueError("CV thresholds must be finite.")
        if self.low_cv < 0 or self.moderate_cv <= self.low_cv:
            raise ValueError("Require 0 <= low_cv < moderate_cv.")

    def classify(self, coefficient_of_variation):
        cv = float(coefficient_of_variation)
        if not np.isfinite(cv) or cv < 0:
            raise ValueError("Coefficient of variation must be finite and non-negative.")
        if cv < self.low_cv:
            return "Low"
        if cv < self.moderate_cv:
            return "Moderate"
        return "High"

    def assess_revenue(self, revenues):
        values = np.asarray(revenues, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("At least two revenue outcomes are required.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Revenue outcomes must be finite and non-negative.")
        mean = statistics.mean(values.tolist())
        median = statistics.median(values.tolist())
        variance = statistics.variance(values.tolist())
        standard_deviation = statistics.stdev(values.tolist())
        cv = standard_deviation / mean if mean > 0 else 0.0
        var_5_revenue = float(np.percentile(values, 5))
        return {
            "mean": mean,
            "median": median,
            "variance": variance,
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": cv,
            "risk_class": self.classify(cv),
            "var_5_revenue": var_5_revenue,
            "loss_var_95_vs_mean": mean - var_5_revenue,
        }