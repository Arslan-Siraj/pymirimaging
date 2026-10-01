class SpectralSmoothing:
    SUPPORTED = {"None", "SavitzkyGolay", "Gaussian"}

    @staticmethod
    def apply(spectra, strategy="None", half_window_size=2):
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)
        if strategy == "None":
            return spectra
        raise NotImplementedError(
            f"Spectral smoothing '{strategy}' is not implemented yet."
        )
