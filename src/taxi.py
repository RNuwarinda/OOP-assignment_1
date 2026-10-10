from abc import ABC, abstractmethod
import statistics

import numpy as np


class Route:
    def __init__(self, name, passenger_counts, fare_ugx):
        self.name = str(name)
        self.passenger_counts = np.asarray(passenger_counts, dtype=float)
        self.fare_ugx = float(fare_ugx)
        if self.passenger_counts.ndim != 1 or len(self.passenger_counts) < 1:
            raise ValueError("Passenger counts must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(self.passenger_counts)) or np.any(self.passenger_counts < 0):
            raise ValueError("Passenger counts must be finite and non-negative.")
        if not np.isfinite(self.fare_ugx) or self.fare_ugx < 0:
            raise ValueError("Fare must be finite and non-negative.")

    def daily_revenue(self):
        return self.passenger_counts * self.fare_ugx

    def total_revenue(self):
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self):
        values = self.passenger_counts.tolist()
        result = {
            "mean_passengers": statistics.mean(values),
            "median_passengers": statistics.median(values),
            "total_revenue_ugx": self.total_revenue(),
        }
        if len(values) > 1:
            result["variance_passengers"] = statistics.variance(values)
            result["standard_deviation_passengers"] = statistics.stdev(values)
        else:
            result["variance_passengers"] = 0.0
            result["standard_deviation_passengers"] = 0.0
        return result

    def __len__(self):
        return len(self.passenger_counts)

    def __repr__(self):
        return (
            f"Route(name={self.name!r}, observations={len(self)}, "
            f"fare_ugx={self.fare_ugx:.0f})"
        )


class Forecaster(ABC):
    def fit(self, observations):
        values = np.asarray(observations, dtype=float)
        if values.ndim != 1 or len(values) < 1:
            raise ValueError("Training observations must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Training observations must be finite and non-negative.")
        self.observations = values.copy()
        self.fitted = True
        return self

    @abstractmethod
    def predict(self, horizon=1):
        raise NotImplementedError

    def _validate_horizon(self, horizon):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")


class MovingAverageForecaster(Forecaster):
    def __init__(self, window=3):
        if not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.window = int(window)

    def predict(self, horizon=1):
        self._validate_horizon(horizon)
        history = self.observations.tolist()
        predictions = []
        for _ in range(horizon):
            prediction = float(np.mean(history[-self.window:]))
            predictions.append(prediction)
            history.append(prediction)
        return np.asarray(predictions)


class ExponentialSmoothingForecaster(Forecaster):
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in the range (0, 1].")

    def fit(self, observations):
        super().fit(observations)
        level = float(self.observations[0])
        for observation in self.observations[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        self.level = level
        return self

    def predict(self, horizon=1):
        self._validate_horizon(horizon)
        return np.full(horizon, self.level, dtype=float)


class LinearTrendForecaster(Forecaster):
    def fit(self, observations):
        super().fit(observations)
        time = np.arange(1, len(self.observations) + 1, dtype=float)
        self.coefficients = np.polyfit(time, self.observations, 1)
        return self

    def predict(self, horizon=1):
        self._validate_horizon(horizon)
        future_time = np.arange(
            len(self.observations) + 1,
            len(self.observations) + horizon + 1,
            dtype=float,
        )
        return np.maximum(0, np.polyval(self.coefficients, future_time))


class SeasonalNaiveForecaster(Forecaster):
    def __init__(self, period=7):
        if not isinstance(period, (int, np.integer)) or period < 1:
            raise ValueError("period must be a positive integer.")
        self.period = int(period)

    def fit(self, observations):
        super().fit(observations)
        if len(self.observations) < self.period:
            raise ValueError("Training data must include at least one full seasonal period.")
        return self

    def predict(self, horizon=1):
        self._validate_horizon(horizon)
        history = self.observations.tolist()
        predictions = []
        for _ in range(horizon):
            prediction = float(history[-self.period])
            predictions.append(prediction)
            history.append(prediction)
        return np.asarray(predictions)