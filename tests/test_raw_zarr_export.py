import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
zarr = pytest.importorskip("zarr")

import pymirimaging as mi


def test_raw_zarr_selected_channels_to_nrrd(tmp_path):
    root = zarr.open(str(tmp_path / "0.zarr"), mode="w")

    cube = np.zeros((3, 4, 3), dtype=np.float32)
    cube[:, :, 0] = 1.0
    cube[:, :, 1] = 2.0
    cube[:, :, 2] = 3.0

    root.create_dataset("hypercube", data=cube, chunks=(3, 4, 1))
    root.create_dataset(
        "wvnm",
        data=np.asarray([1600.0, 1650.0, 1690.0], dtype=np.float64),
    )

    # Geometry can be stored directly in Zarr, so INFO.txt is not required.
    root.attrs["spacing"] = [0.004, 0.005, 0.01]
    root.attrs["origin"] = [0.0, 0.0, 0.0]
    root.attrs["direction"] = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    root.attrs["orientation"] = "raw_mir"
    root.attrs["modality"] = "MIR"
    root.attrs["spectral_unit"] = "cm^-1"

    reader = mi.ZarrSpectrumReader(tmp_path / "0.zarr")

    output = tmp_path / "selected.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1650.0, 1690.0],
        selection="Exact",
        rotate=90,
    )

    image = sitk.ReadImage(str(output))
    data = sitk.GetArrayFromImage(image)

    # Raw MIR export rotates Y/X once before M2aia NRRD export.
    assert data.shape == (1, 4, 3, 2)
    assert image.GetMetaData("Type") == "1650,1690"
    assert image.GetMetaData("Modality") == "MIR"
    assert image.GetMetaData("SpectralUnit") == "cm^-1"

    # Rotation swaps X/Y spacing.
    assert np.allclose(image.GetSpacing(), [0.005, 0.004, 0.01])
