import numpy as np

from pymirimaging.MIRprocessing import (
    SpectrumNormalization,
    SpectralSelection,
    SpectralPooling,
)


def test_max_normalization():
    spectra = np.asarray(
        [[1, 2, 4], [2, 4, 8]],
        dtype=np.float32,
    )
    axis = np.asarray([1000, 1001, 1002], dtype=float)

    result = SpectrumNormalization.apply(
        spectra, axis, "Max"
    )

    assert np.allclose(np.max(result, axis=1), 1.0)


def test_exact_selection():
    axis = np.asarray([1650.0, 1690.0])
    indices = SpectralSelection.select_indices(
        axis, 1650.0, strategy="Exact"
    )
    assert np.array_equal(indices, [0])


def test_window_pooling():
    selected = np.asarray(
        [[1.0, 3.0], [2.0, 4.0]],
        dtype=np.float32,
    )
    values = SpectralPooling.apply(
        selected,
        [1650.0, 1690.0],
        1670.0,
        strategy="Mean",
    )
    assert np.allclose(values, [2.0, 3.0])
