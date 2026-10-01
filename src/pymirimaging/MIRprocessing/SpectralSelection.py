import numpy as np


class SpectralSelection:
    """Select MIR channels by nearest, exact, or tolerance-window lookup."""

    SUPPORTED = {"Nearest", "Exact", "Window"}

    @staticmethod
    def _validate_axis(x_axis):
        x_axis = np.asarray(x_axis, dtype=np.float64)
        if x_axis.ndim != 1 or x_axis.size == 0:
            raise ValueError("x_axis must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(x_axis)):
            raise ValueError("x_axis contains non-finite values.")
        return x_axis

    @staticmethod
    def nearest_index(x_axis, center):
        x_axis = SpectralSelection._validate_axis(x_axis)
        center = float(center)
        if not np.isfinite(center):
            raise ValueError("center must be finite.")
        return int(np.argmin(np.abs(x_axis - center)))

    @staticmethod
    def nearest_value(x_axis, center):
        x_axis = SpectralSelection._validate_axis(x_axis)
        return float(x_axis[SpectralSelection.nearest_index(x_axis, center)])

    @staticmethod
    def select_indices(x_axis, center, strategy="Nearest", tolerance=0.0):
        x_axis = SpectralSelection._validate_axis(x_axis)
        center = float(center)
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)

        if strategy not in SpectralSelection.SUPPORTED:
            raise ValueError(
                f"Unsupported spectral selection '{strategy}'. "
                f"Supported: {sorted(SpectralSelection.SUPPORTED)}"
            )

        nearest = SpectralSelection.nearest_index(x_axis, center)

        if strategy == "Nearest":
            return np.asarray([nearest], dtype=np.int64)

        if strategy == "Exact":
            scale = max(1.0, abs(center), float(np.max(np.abs(x_axis))))
            atol = np.finfo(np.float64).eps * scale * 8.0
            indices = np.flatnonzero(
                np.isclose(x_axis, center, rtol=0.0, atol=atol)
            )
            if indices.size == 0:
                raise ValueError(
                    f"No exact spectral channel found at {center}. "
                    f"Nearest channel is {x_axis[nearest]}."
                )
            return indices.astype(np.int64, copy=False)

        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerance must be a finite value >= 0.")

        indices = np.flatnonzero(
            (x_axis >= center - tolerance)
            & (x_axis <= center + tolerance)
        )
        if indices.size == 0:
            raise ValueError(
                f"No spectral channel found in "
                f"[{center - tolerance}, {center + tolerance}]. "
                f"Nearest channel is {x_axis[nearest]}."
            )
        return indices.astype(np.int64, copy=False)
