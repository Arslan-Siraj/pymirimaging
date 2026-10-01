from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import zarr

from ._helpers import format_wavenumbers, json_safe
from .MIRprocessing import SpectralSelection


class NRRDSpectrumReader:
    """
    Read MIR vector NRRD files for M2aia interchange.

    This reader intentionally focuses on import/inspection/export-to-Zarr.
    MIR processing is performed with ZarrSpectrumReader.
    """

    def __init__(self, nrrd_path, x_axis=None):
        self.nrrd_path = str(nrrd_path)
        self.image = None
        self.data = None
        self.x_axis = None
        self.Load(x_axis=x_axis)

    def Load(self, x_axis=None):
        path = Path(self.nrrd_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self.image = sitk.ReadImage(str(path))
        components = self.image.GetNumberOfComponentsPerPixel()

        data = sitk.GetArrayFromImage(self.image)

        if self.image.GetDimension() == 2:
            if data.ndim == 2 and components == 1:
                data = data[..., np.newaxis]
            if data.ndim != 3:
                raise ValueError(
                    "Expected 2D vector NRRD array [y,x,channels]."
                )
            data = data[np.newaxis, ...]
        elif self.image.GetDimension() == 3:
            if data.ndim == 3 and components == 1:
                data = data[..., np.newaxis]
            if data.ndim != 4:
                raise ValueError(
                    "Expected 3D vector NRRD array [z,y,x,channels]."
                )
        else:
            raise ValueError(
                "Expected a 2D or 3D MIR vector NRRD."
            )

        self.data = np.asarray(data)
        self.depth_z, self.height, self.width, self.depth = (
            self.data.shape
        )
        self.number_of_spectra = (
            self.depth_z * self.height * self.width
        )

        if x_axis is None:
            x_axis = self._read_x_axis()

        self.x_axis = np.asarray(x_axis, dtype=np.float64)
        if self.x_axis.ndim != 1:
            raise ValueError("x_axis must be one-dimensional.")
        if self.x_axis.size != self.depth:
            raise ValueError(
                f"x_axis has {self.x_axis.size} values, "
                f"but NRRD has {self.depth} channels."
            )

    def _read_x_axis(self):
        keys = {
            key.lower(): key
            for key in self.image.GetMetaDataKeys()
        }

        for candidate in (
            "m2aia_xaxis",
            "m2aia.xaxis",
            "wavenumbers",
            "wavenumber",
            "type",
        ):
            if candidate not in keys:
                continue

            raw = self.image.GetMetaData(keys[candidate])
            try:
                values = [
                    float(v.strip())
                    for v in raw.replace(";", ",").split(",")
                    if v.strip()
                ]
            except ValueError:
                continue

            if len(values) == self.depth:
                return np.asarray(values, dtype=np.float64)

            if len(values) == 2 and self.depth > 2:
                return np.linspace(
                    values[0],
                    values[1],
                    self.depth,
                    dtype=np.float64,
                )

        raise ValueError(
            "Could not determine MIR spectral axis from NRRD metadata. "
            "Pass x_axis explicitly."
        )

    def path(self):
        return Path(self.nrrd_path)

    def dir(self):
        return self.path().parent

    def name(self):
        return self.path().name

    def GetImageName(self):
        return self.path().stem

    def GetModality(self):
        if self.image.HasMetaDataKey("Modality"):
            return self.image.GetMetaData("Modality")
        return "MIR"

    def GetSpectralUnit(self):
        if self.image.HasMetaDataKey("SpectralUnit"):
            return self.image.GetMetaData("SpectralUnit")
        return "cm^-1"

    def GetSpectrumType(self):
        return "ContinuousProfile"

    def GetShape(self):
        return np.asarray(
            [self.width, self.height, self.depth_z],
            dtype=np.int32,
        )

    def GetSpacing(self):
        return np.asarray(
            self.image.GetSpacing(),
            dtype=np.float64,
        )

    def GetOrigin(self):
        return np.asarray(
            self.image.GetOrigin(),
            dtype=np.float64,
        )

    def GetDirection(self):
        return np.asarray(
            self.image.GetDirection(),
            dtype=np.float64,
        )

    def GetXAxis(self):
        return self.x_axis.copy()

    def GetXAxisDepth(self):
        return self.depth

    def GetNumberOfSpectra(self):
        return self.number_of_spectra

    def GetSpectrumPosition(self, index):
        index = int(index)
        if index < 0 or index >= self.number_of_spectra:
            raise IndexError(index)
        z, y, x = np.unravel_index(
            index,
            (self.depth_z, self.height, self.width),
        )
        return np.asarray([x, y, z], dtype=np.int32)

    def GetSpectrum(self, index) -> List[np.ndarray]:
        index = int(index)
        if index < 0 or index >= self.number_of_spectra:
            raise IndexError(index)
        spectrum = self.data.reshape(
            -1, self.depth
        )[index]
        return [
            self.x_axis.astype(np.float32, copy=False),
            np.asarray(spectrum, dtype=np.float32),
        ]

    def GetArray(
        self,
        center,
        squeeze=False,
        selection="Nearest",
        tolerance=0.0,
    ):
        indices = SpectralSelection.select_indices(
            self.x_axis,
            center,
            strategy=selection,
            tolerance=tolerance,
        )

        if len(indices) != 1:
            raise ValueError(
                "NRRDSpectrumReader.GetArray is raw import access only. "
                "Use one Exact/Nearest channel, or convert to Zarr for "
                "window/pooling/normalization processing."
            )

        result = np.asarray(
            self.data[..., int(indices[0])],
            dtype=np.float32,
        )
        return np.squeeze(result) if squeeze else result

    def GetMaskArray(self):
        valid = np.all(
            np.isfinite(self.data),
            axis=-1,
        )
        return valid.astype(np.ushort)

    def GetMetaData(self) -> Dict[str, str]:
        return {
            key: self.image.GetMetaData(key)
            for key in self.image.GetMetaDataKeys()
        }

    def WriteZarr(
        self,
        output_path,
        overwrite=False,
        squeeze_single_z=True,
    ):
        """
        Convert an MIR NRRD to the Zarr working representation while
        preserving spectral axis, geometry, and source metadata.
        """
        output_path = Path(output_path)
        out = zarr.open(
            str(output_path),
            mode="w" if overwrite else "w-",
        )

        if squeeze_single_z and self.depth_z == 1:
            cube = self.data[0]
        else:
            cube = self.data

        out.create_dataset(
            "hypercube",
            data=cube,
            chunks=True,
        )
        out.create_dataset(
            "wvnm",
            data=self.x_axis.astype(np.float64),
        )

        mask = self.GetMaskArray()
        if np.any(mask == 0):
            if squeeze_single_z and self.depth_z == 1:
                mask = mask[0]
            out.create_dataset(
                "mask",
                data=mask.astype(np.uint8),
                chunks=True,
            )

        out.attrs["spacing"] = self.GetSpacing().tolist()
        out.attrs["origin"] = self.GetOrigin().tolist()
        out.attrs["direction"] = self.GetDirection().tolist()
        out.attrs["modality"] = self.GetModality()
        out.attrs["spectral_unit"] = self.GetSpectralUnit()
        out.attrs["spatial_unit"] = "mm"
        out.attrs["orientation"] = "m2aia_nrrd"
        out.attrs["source_format"] = "NRRD"
        out.attrs["source_metadata"] = json_safe(
            self.GetMetaData()
        )
        out.attrs["processing_history"] = []

        return output_path
