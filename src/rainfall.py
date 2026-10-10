import numpy as np


MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


class Region:
    def __init__(self, name, monthly_rainfall_mm, months=MONTHS):
        self.name = str(name)
        self.months = tuple(months)
        self.rainfall_mm = np.asarray(monthly_rainfall_mm, dtype=float)
        if self.rainfall_mm.shape != (12,):
            raise ValueError("A region must have exactly 12 monthly rainfall values.")
        if len(self.months) != 12:
            raise ValueError("Exactly 12 month labels are required.")
        if not np.all(np.isfinite(self.rainfall_mm)) or np.any(self.rainfall_mm < 0):
            raise ValueError("Rainfall values must be finite and non-negative.")

    def annual_total(self):
        return float(np.sum(self.rainfall_mm))

    def mean_monthly_rainfall(self):
        return float(np.mean(self.rainfall_mm))

    def wettest_month(self):
        index = int(np.argmax(self.rainfall_mm))
        return self.months[index], float(self.rainfall_mm[index])

    def driest_month(self):
        index = int(np.argmin(self.rainfall_mm))
        return self.months[index], float(self.rainfall_mm[index])

    def coefficient_of_variation(self):
        mean = self.mean_monthly_rainfall()
        return float(np.std(self.rainfall_mm, ddof=0) / mean) if mean else 0.0

    def __len__(self):
        return len(self.rainfall_mm)

    def __repr__(self):
        return f"Region(name={self.name!r}, annual_rainfall_mm={self.annual_total():.1f})"


class CropRule:
    def __init__(self, crop_name, minimum_monthly_mm, maximum_monthly_mm, source):
        self.crop_name = str(crop_name)
        self.minimum_monthly_mm = float(minimum_monthly_mm)
        self.maximum_monthly_mm = float(maximum_monthly_mm)
        self.source = str(source)
        if not np.all(np.isfinite([
            self.minimum_monthly_mm, self.maximum_monthly_mm
        ])):
            raise ValueError("Crop rainfall thresholds must be finite.")
        if self.minimum_monthly_mm < 0 or self.maximum_monthly_mm < self.minimum_monthly_mm:
            raise ValueError("Crop rainfall thresholds must be non-negative and ordered.")
        if not self.source:
            raise ValueError("A source or derivation must be provided for crop thresholds.")

    def classify_month(self, rainfall_mm):
        rainfall_mm = float(rainfall_mm)
        if not np.isfinite(rainfall_mm) or rainfall_mm < 0:
            raise ValueError("Monthly rainfall must be finite and non-negative.")
        if rainfall_mm < self.minimum_monthly_mm:
            return f"Drought risk for {self.crop_name}"
        if rainfall_mm > self.maximum_monthly_mm:
            return f"Waterlogging risk for {self.crop_name}"
        return f"Good for {self.crop_name}"