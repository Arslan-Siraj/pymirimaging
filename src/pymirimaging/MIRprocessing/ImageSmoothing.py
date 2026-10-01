class ImageSmoothing:
    SUPPORTED = {"None", "Median", "Gaussian"}

    @staticmethod
    def apply(array, strategy="None"):
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)
        if strategy == "None":
            return array
        raise NotImplementedError(
            f"Image smoothing '{strategy}' is not implemented yet."
        )
