class BaselineCorrection:
    SUPPORTED = {"None", "TopHat", "Median"}

    @staticmethod
    def apply(spectra, strategy="None", half_window_size=50):
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)
        if strategy == "None":
            return spectra
        raise NotImplementedError(
            f"Baseline correction '{strategy}' is not implemented yet."
        )
