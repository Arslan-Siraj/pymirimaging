import numpy as np


class SpectrumNormalization:
    """Normalize each MIR spectrum independently."""

    SUPPORTED = {"None", "TIC", "Sum", "Mean", "Max", "RMS"}

    @staticmethod
    def factor(spectra, x_axis, strategy="None"):
        spectra = np.asarray(spectra, dtype=np.float32)
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)

        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [number_of_spectra, channels].")

        if strategy == "None":
            return np.ones((spectra.shape[0], 1), dtype=np.float32)

        if strategy == "TIC":
            if spectra.shape[1] == 1:
                factors = spectra[:, 0]
            else:
                x_axis = np.asarray(x_axis, dtype=np.float64)
                dx = np.diff(x_axis)
                factors = np.sum(
                    0.5 * (spectra[:, :-1] + spectra[:, 1:]) * dx[None, :],
                    axis=1,
                )
                factors = np.abs(factors)
        elif strategy == "Sum":
            factors = np.sum(spectra, axis=1)
        elif strategy == "Mean":
            factors = np.mean(spectra, axis=1)
        elif strategy == "Max":
            factors = np.max(spectra, axis=1)
        elif strategy == "RMS":
            factors = np.sqrt(
                np.mean(np.square(spectra, dtype=np.float64), axis=1)
            )
        else:
            raise ValueError(
                f"Unsupported normalization '{strategy}'. "
                f"Supported: {sorted(SpectrumNormalization.SUPPORTED)}"
            )

        factors = np.asarray(factors, dtype=np.float32)
        factors[~np.isfinite(factors)] = 1.0
        factors[factors == 0] = 1.0
        return factors[:, None]

    @staticmethod
    def apply(spectra, x_axis, strategy="None"):
        spectra = np.asarray(spectra, dtype=np.float32)
        return (
            spectra
            / SpectrumNormalization.factor(spectra, x_axis, strategy)
        ).astype(np.float32, copy=False)
