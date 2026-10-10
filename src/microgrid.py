import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls


class MicroGrid:
    system_size = 2
    default_coefficients = np.array([[3.0, 2.0], [4.0, 1.0]])

    def __init__(self, coefficients=None):
        matrix = self.default_coefficients if coefficients is None else coefficients
        self.coefficients = np.asarray(matrix, dtype=float).copy()
        if self.coefficients.shape != (self.system_size, self.system_size):
            raise ValueError(
                f"The coefficient matrix must have shape "
                f"({self.system_size}, {self.system_size})."
            )
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("The coefficient matrix must contain only finite values.")
        self.determinant = float(np.linalg.det(self.coefficients))
        self.condition_number = float(np.linalg.cond(self.coefficients))

    @property
    def is_well_posed(self):
        return (
            np.isfinite(self.condition_number)
            and self.determinant != 0.0
            and self.condition_number < 1e12
        )

    def _validate_demands(self, demands):
        values = np.asarray(demands, dtype=float)
        if values.shape != (self.system_size,):
            raise ValueError(f"Expected {self.system_size} demand values.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demand values must be finite and non-negative.")
        return values

    def _solve(self, demands):
        if not self.is_well_posed:
            raise ValueError(
                "The system is singular or too ill-conditioned to solve reliably."
            )
        return solve(self.coefficients, demands, assume_a="gen")

    def solve_day(self, d1, d2):
        """Return the exact solar and battery solution for one day's demands."""
        demands = self._validate_demands([d1, d2])
        return self._solve(demands)

    def solve_day_nonnegative(self, d1, d2):
        """Return an NNLS dispatch and residual norm for physically valid output."""
        demands = self._validate_demands([d1, d2])
        if not self.is_well_posed:
            raise ValueError(
                "The system is singular or too ill-conditioned to solve reliably."
            )
        dispatch, residual_norm = nnls(self.coefficients, demands)
        return dispatch, float(residual_norm)

    @staticmethod
    def is_physically_feasible(dispatch, tolerance=1e-9):
        values = np.asarray(dispatch, dtype=float)
        return bool(np.all(np.isfinite(values)) and np.all(values >= -tolerance))


class HybridMicroGrid(MicroGrid):
    system_size = 3
    default_coefficients = np.array(
        [
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
        ]
    )

    def solve_day(self, d1, d2, d3):
        """Solve demands with solar, battery, and diesel variables."""
        demands = self._validate_demands([d1, d2, d3])
        return self._solve(demands)

    def solve_day_nonnegative(self, d1, d2, d3):
        demands = self._validate_demands([d1, d2, d3])
        if not self.is_well_posed:
            raise ValueError(
                "The system is singular or too ill-conditioned to solve reliably."
            )
        dispatch, residual_norm = nnls(self.coefficients, demands)
        return dispatch, float(residual_norm)