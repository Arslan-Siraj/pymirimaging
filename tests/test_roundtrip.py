from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
zarr = pytest.importorskip("zarr")

import pymirimaging as m2


def _make_test_nrrd(path):
    data = np.zeros((1, 3, 4, 2), dtype=np.float32)
    data[..., 0] = 10
    data[..., 1] = 20

    image = sitk.GetImageFromArray(data, isVector=True)
    image.SetSpacing((0.004, 0.005, 0.01))
    image.SetOrigin((1.0, 2.0, 0.0))
    image.SetDirection(
        (1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0)
    )
    image.SetMetaData("Type", "1650,1690")
    image.SetMetaData("Modality", "MIR")
    image.SetMetaData("SpectralUnit", "cm^-1")
    sitk.WriteImage(image, str(path))


def test_nrrd_to_zarr_to_nrrd(tmp_path):
    source = tmp_path / "source.nrrd"
    _make_test_nrrd(source)

    n = m2.NRRDSpectrumReader(source)
    zarr_path = tmp_path / "mir.zarr"
    n.WriteZarr(zarr_path)

    z = m2.ZarrSpectrumReader(zarr_path)
    assert np.allclose(z.GetXAxis(), [1650, 1690])
    assert np.allclose(
        z.GetSpacing(),
        [0.004, 0.005, 0.01],
    )

    output = tmp_path / "roundtrip.nrrd"
    z.WriteNRRD(output)

    result = m2.NRRDSpectrumReader(output)

    assert np.allclose(
        result.GetXAxis(),
        n.GetXAxis(),
    )
    assert np.allclose(
        result.GetSpacing(),
        n.GetSpacing(),
    )
    assert np.allclose(
        result.GetOrigin(),
        n.GetOrigin(),
    )
    assert np.allclose(
        result.GetDirection(),
        n.GetDirection(),
    )
    assert np.allclose(
        result.data,
        n.data,
    )
