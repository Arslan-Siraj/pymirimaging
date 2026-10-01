class IntensityTransformation:
    SUPPORTED = {"None", "SquareRoot", "Log2", "Log10"}

    @staticmethod
    def apply(spectra, strategy="None"):
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)
        if strategy == "None":
            return spectra
        raise NotImplementedError(
            f"Intensity transformation '{strategy}' is not implemented yet."
        )
