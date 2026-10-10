from abc import ABC, abstractmethod

import numpy as np


class DistrictPopulation:
    def __init__(self, district_name, years, population):
        self.district_name = str(district_name)
        self.years = np.asarray(years, dtype=int)
        self.population = np.asarray(population, dtype=float)

        if self.years.ndim != 1 or self.population.ndim != 1:
            raise ValueError("Years and population must be one-dimensional arrays.")
        if len(self.years) == 0 or len(self.years) != len(self.population):
            raise ValueError("Years and population must have the same non-zero length.")
        if np.any(np.diff(self.years) <= 0):
            raise ValueError("Years must be strictly increasing.")
        if not np.all(np.isfinite(self.population)) or np.any(self.population < 0):
            raise ValueError("Population values must be finite and non-negative.")

    def __repr__(self):
        return (
            f"DistrictPopulation(district_name={self.district_name!r}, "
            f"years={self.years!r}, population={self.population!r})"
        )

    def __len__(self):
        return len(self.population)

    def get_district_name(self):
        return self.district_name

    def get_population(self):
        return self.population


class Forecaster(ABC):
    def fit(self, years, population):
        self.years = np.asarray(years, dtype=float)
        self.population = np.asarray(population, dtype=float)
        if self.years.ndim != 1 or self.population.ndim != 1:
            raise ValueError("Training years and population must be one-dimensional.")
        if len(self.years) == 0 or len(self.years) != len(self.population):
            raise ValueError("Training years and population must have equal non-zero length.")
        if np.any(np.diff(self.years) <= 0) or np.any(self.population < 0):
            raise ValueError("Training years must increase and population cannot be negative.")
        self.last_year = self.years[-1]
        self.fitted = True
        return self

    @abstractmethod
    def predict(self, horizon):
        raise NotImplementedError

    @property
    @abstractmethod
    def fitted_values(self):
        raise NotImplementedError

    def _future_years(self, horizon):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        return np.arange(self.last_year + 1, self.last_year + horizon + 1)


class LinearTrendForecaster(Forecaster):
    def fit(self, years, population):
        super().fit(years, population)
        self.coefficients = np.polyfit(self.years, self.population, 1)
        return self

    @property
    def fitted_values(self):
        return np.polyval(self.coefficients, self.years)

    def predict(self, horizon):
        future_years = self._future_years(horizon)
        return np.polyval(self.coefficients, future_years)


class CAGRForecaster(Forecaster):
    def fit(self, years, population):
        super().fit(years, population)
        if self.population[0] <= 0:
            raise ValueError("CAGR forecasting requires a positive first population value.")
        elapsed_years = self.years[-1] - self.years[0]
        self.annual_rate = (self.population[-1] / self.population[0]) ** (
            1 / elapsed_years
        ) - 1 if elapsed_years > 0 else 0.0
        return self

    @property
    def fitted_values(self):
        elapsed = self.years - self.years[0]
        return self.population[0] * (1 + self.annual_rate) ** elapsed

    def predict(self, horizon):
        future_years = self._future_years(horizon)
        elapsed = future_years - self.last_year
        return self.population[-1] * (1 + self.annual_rate) ** elapsed


class FibonacciRatioForecaster(Forecaster):
    @staticmethod
    def _ratios(count):
        terms = [1, 1]
        while len(terms) <= count + 2:
            terms.append(terms[-1] + terms[-2])
        return np.asarray(
            [terms[index + 1] / terms[index] for index in range(2, count + 2)],
            dtype=float,
        )

    @property
    def fitted_values(self):
        ratios = self._ratios(len(self.population) - 1)
        return np.concatenate(
            ([self.population[0]], self.population[0] * np.cumprod(ratios))
        )

    def predict(self, horizon):
        self._future_years(horizon)
        ratios = self._ratios(horizon)
        return self.population[-1] * np.cumprod(ratios)
