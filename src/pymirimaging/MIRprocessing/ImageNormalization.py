import numpy as np


class ImageNormalization:
    """Normalize an extracted spatial image."""

    SUPPORTED = {"None", "MinMax"}

    @staticmethod
    def apply(array, strategy="None", valid_mask=None):
        strategy = strategy.value if hasattr(strategy, "value") else str(strategy)
        array = np.asarray(array)

        if strategy == "None":
            return array

        if strategy != "MinMax":
            raise ValueError(
                f"Unsupported image normalization '{strategy}'. "
                f"Supported: {sorted(ImageNormalization.SUPPORTED)}"
            )

        result = np.asarray(array, dtype=np.float32).copy()

        if valid_mask is None:
            valid = np.isfinite(result)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != result.shape:
                raise ValueError("valid_mask must have the same shape as array.")
            valid &= np.isfinite(result)

        values = result[valid]
        if values.size == 0:
            return result

        minimum = float(np.min(values))
        maximum = float(np.max(values))
        denominator = maximum - minimum

        if denominator == 0:
            result[valid] = 0.0
        else:
            result[valid] = (values - minimum) / denominator

        return result
