from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import zarr

from ._helpers import (
    value_of,
    read_pixel_sizes,
    format_wavenumbers,
    json_safe,
)
from .MIRprocessing import (
    SpectrumNormalization,
    ImageNormalization,
    SpectralPooling,
    SpectralSelection,
)


class ZarrSpectrumReader:
    """
    pyM2aia-style MIR reader for Zarr datasets.

    Expected datasets:
        hypercube : (Y, X, C) or (Z, Y, X, C)
        wvnm      : (C,)

    Processing is performed directly from the Zarr-backed hypercube.
    """

    _NORMALIZATIONS = SpectrumNormalization.SUPPORTED
    _POOLING = SpectralPooling.SUPPORTED
    _SPECTRAL_SELECTIONS = SpectralSelection.SUPPORTED
    _IMAGE_NORMALIZATIONS = ImageNormalization.SUPPORTED

    def __init__(
        self,
        zarr_path,
        info_path=None,
        pz=None,
        normalization="None",
        pooling="Maximum",
        image_normalization="None",
        spectral_selection="Nearest",
        tolerance=0.0,
        chunk_rows=64,
    ):
        self.zarr_path = str(zarr_path)
        self.info_path = info_path
        self.pz = pz
        self.chunk_rows = max(1, int(chunk_rows))

        self.root = None
        self.data = None
        self.x_axis = None
        self._valid_mask = None

        self.normalization = "None"
        self.pooling = "Maximum"
        self.image_normalization = "None"
        self.spectral_selection = "Nearest"
        self.tolerance = np.float32(0.0)

        self.Load()
        self.SetNormalization(normalization)
        self.SetPooling(pooling)
        self.SetImageNormalization(image_normalization)
        self.SetSpectralSelection(spectral_selection)
        self.SetTolerance(tolerance)

    def Load(self):
        path = Path(self.zarr_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self.root = zarr.open(str(path), mode="r")

        if "hypercube" not in self.root or "wvnm" not in self.root:
            raise ValueError(
                "Expected Zarr datasets 'hypercube' and 'wvnm'."
            )

        self.data = self.root["hypercube"]
        self.x_axis = np.asarray(self.root["wvnm"][:], dtype=np.float64)

        if self.data.ndim not in (3, 4):
            raise ValueError(
                "Expected hypercube shape (Y, X, C) or (Z, Y, X, C), "
                f"got {self.data.shape}."
            )

        if self.x_axis.ndim != 1:
            raise ValueError("wvnm must be one-dimensional.")

        if self.data.shape[-1] != self.x_axis.size:
            raise ValueError(
                "hypercube spectral depth does not match wvnm length."
            )

        if not np.all(np.isfinite(self.x_axis)):
            raise ValueError("wvnm contains non-finite values.")

        if self.data.ndim == 3:
            self.depth_z = 1
            self.height, self.width, self.depth = self.data.shape
        else:
            self.depth_z, self.height, self.width, self.depth = self.data.shape

        self.number_of_spectra = (
            self.depth_z * self.height * self.width
        )

        self._load_geometry()
        self._load_mask()

    def _load_geometry(self):
        attrs = dict(self.root.attrs)

        spacing = attrs.get("spacing")
        origin = attrs.get("origin")
        direction = attrs.get("direction")

        if spacing is None:
            try:
                px, py, info = read_pixel_sizes(
                    self.zarr_path, self.info_path
                )
                self.info_path = str(info)
                spacing = [px / 1000.0, py / 1000.0]
                if self.pz is not None:
                    pz = float(self.pz)
                    if not np.isfinite(pz) or pz <= 0:
                        raise ValueError(
                            "pz must be a positive finite value in micrometres."
                        )
                    spacing.append(pz / 1000.0)
            except FileNotFoundError:
                spacing = None

        self._spacing = (
            None if spacing is None
            else np.asarray(spacing, dtype=np.float64)
        )

        if origin is None and self._spacing is not None:
            origin = [0.0] * len(self._spacing)
        self._origin = (
            None if origin is None
            else np.asarray(origin, dtype=np.float64)
        )

        if direction is None and self._spacing is not None:
            n = len(self._spacing)
            direction = np.eye(n, dtype=np.float64).reshape(-1)
        self._direction = (
            None if direction is None
            else np.asarray(direction, dtype=np.float64)
        )

    def _load_mask(self):
        if "mask" in self.root:
            mask = np.asarray(self.root["mask"][:], dtype=bool)
            expected = (
                (self.height, self.width)
                if self.depth_z == 1
                else (self.depth_z, self.height, self.width)
            )
            if mask.shape != expected:
                raise ValueError(
                    f"mask shape {mask.shape} does not match {expected}."
                )
            self._valid_mask = mask
        else:
            self._valid_mask = None

    def path(self):
        return Path(self.zarr_path)

    def dir(self):
        return self.path().parent

    def name(self):
        return self.path().name

    def GetImageName(self):
        return self.path().stem

    def CheckHandle(self):
        if self.root is None or self.data is None:
            raise ReferenceError("Zarr reader is not initialized.")

    def GetModality(self):
        return str(self.root.attrs.get("modality", "MIR"))

    def GetSpectralUnit(self):
        return str(self.root.attrs.get("spectral_unit", "cm^-1"))

    def GetSpectrumType(self):
        return str(
            self.root.attrs.get(
                "spectrum_type", "ContinuousProfile"
            )
        )

    def GetShape(self):
        return np.asarray(
            [self.width, self.height, self.depth_z],
            dtype=np.int32,
        )

    def GetSpacing(self):
        if self._spacing is None:
            raise ValueError(
                "Spatial spacing is unavailable. Provide info_path/pz or "
                "store 'spacing' in Zarr attributes."
            )
        return self._spacing.copy()

    def GetOrigin(self):
        if self._origin is None:
            raise ValueError("Spatial origin is unavailable.")
        return self._origin.copy()

    def GetDirection(self):
        if self._direction is None:
            raise ValueError("Spatial direction is unavailable.")
        return self._direction.copy()

    def GetXAxis(self):
        return self.x_axis.copy()

    def GetXAxisDepth(self):
        return int(self.depth)

    def GetYDataType(self):
        return np.dtype(self.data.dtype).type

    def GetSizeInBytesOfYAxisType(self):
        return np.dtype(self.data.dtype).itemsize

    def GetNumberOfSpectra(self):
        if self._valid_mask is None:
            return int(self.number_of_spectra)
        return int(np.count_nonzero(self._valid_mask))

    def _position_from_linear(self, index):
        index = int(index)
        if index < 0 or index >= self.number_of_spectra:
            raise IndexError(index)
        z, y, x = np.unravel_index(
            index,
            (self.depth_z, self.height, self.width),
        )
        return int(z), int(y), int(x)

    def GetSpectrumPosition(self, index):
        z, y, x = self._position_from_linear(index)
        return np.asarray([x, y, z], dtype=np.int32)

    def _raw_spectrum(self, index):
        z, y, x = self._position_from_linear(index)
        if self.data.ndim == 3:
            return np.asarray(self.data[y, x, :], dtype=np.float32)
        return np.asarray(self.data[z, y, x, :], dtype=np.float32)

    def GetSpectrumDepth(self, index):
        self._position_from_linear(index)
        return self.depth

    def SetNormalization(self, strategy):
        strategy = value_of(strategy)
        if strategy not in self._NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization '{strategy}'. "
                f"Supported: {sorted(self._NORMALIZATIONS)}"
            )
        self.normalization = strategy

    def SetPooling(self, strategy):
        strategy = value_of(strategy)
        if strategy not in self._POOLING:
            raise ValueError(
                f"Unsupported pooling '{strategy}'. "
                f"Supported: {sorted(self._POOLING)}"
            )
        self.pooling = strategy

    def SetImageNormalization(self, strategy):
        strategy = value_of(strategy)
        if strategy not in self._IMAGE_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported image normalization '{strategy}'. "
                f"Supported: {sorted(self._IMAGE_NORMALIZATIONS)}"
            )
        self.image_normalization = strategy

    def SetSpectralSelection(self, strategy):
        strategy = value_of(strategy)
        if strategy not in self._SPECTRAL_SELECTIONS:
            raise ValueError(
                f"Unsupported spectral selection '{strategy}'. "
                f"Supported: {sorted(self._SPECTRAL_SELECTIONS)}"
            )
        self.spectral_selection = strategy

    def GetSpectralSelection(self):
        return self.spectral_selection

    def SetTolerance(self, tolerance):
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerance must be finite and >= 0.")
        self.tolerance = np.float32(tolerance)

    def GetTolerance(self):
        return self.tolerance

    def GetNearestWavenumber(self, center):
        return SpectralSelection.nearest_value(
            self.x_axis, center
        )

    def _select_spectral_indices(
        self, center, tol=None, selection=None
    ):
        strategy = (
            self.spectral_selection
            if selection is None
            else value_of(selection)
        )
        tolerance = self.tolerance if tol is None else float(tol)
        return SpectralSelection.select_indices(
            self.x_axis,
            center,
            strategy=strategy,
            tolerance=tolerance,
        )

    def GetSelectedWavenumbers(
        self, center, tol=None, selection=None
    ):
        indices = self._select_spectral_indices(
            center, tol=tol, selection=selection
        )
        return self.x_axis[indices].copy()

    def _process_spectra(self, spectra):
        return SpectrumNormalization.apply(
            spectra,
            self.x_axis,
            self.normalization,
        )

    def GetSpectrum(self, index) -> List[np.ndarray]:
        raw = self._raw_spectrum(index)[None, :]
        ys = self._process_spectra(raw)[0]
        return [
            self.x_axis.astype(np.float32, copy=False),
            ys.astype(np.float32, copy=False),
        ]

    def GetIntensities(self, index, ys=None):
        values = self.GetSpectrum(index)[1]
        if ys is None:
            return values.copy()
        if ys.dtype != np.float32:
            raise TypeError("ys must have dtype=np.float32.")
        if ys.shape[0] != self.depth:
            ys.resize((self.depth,), refcheck=False)
        ys[:] = values
        return ys

    def GetSpectra(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.zeros((0, self.depth), dtype=np.float32)

        spectra = np.stack(
            [self._raw_spectrum(i) for i in indices],
            axis=0,
        )
        return self._process_spectra(spectra)

    def _iter_blocks(self):
        if self.data.ndim == 3:
            for y0 in range(0, self.height, self.chunk_rows):
                y1 = min(self.height, y0 + self.chunk_rows)
                block = np.asarray(
                    self.data[y0:y1, :, :],
                    dtype=np.float32,
                )
                yield 0, y0, y1, block
        else:
            for z in range(self.depth_z):
                for y0 in range(0, self.height, self.chunk_rows):
                    y1 = min(self.height, y0 + self.chunk_rows)
                    block = np.asarray(
                        self.data[z, y0:y1, :, :],
                        dtype=np.float32,
                    )
                    yield z, y0, y1, block

    def GetArray(
        self,
        center,
        tol=None,
        dtype=np.float32,
        squeeze=False,
        selection=None,
    ):
        if dtype not in (np.float32, np.float64):
            raise TypeError(
                "Image dtype must be np.float32 or np.float64."
            )

        indices = self._select_spectral_indices(
            center, tol=tol, selection=selection
        )

        output = np.zeros(
            (self.depth_z, self.height, self.width),
            dtype=np.float32,
        )

        for z, y0, y1, block in self._iter_blocks():
            spectra = block.reshape(-1, self.depth)
            spectra = self._process_spectra(spectra)
            selected = spectra[:, indices]
            values = SpectralPooling.apply(
                selected,
                self.x_axis[indices],
                center,
                self.pooling,
            )
            output[z, y0:y1, :] = values.reshape(
                y1 - y0, self.width
            )

        mask = self.GetMaskArray().astype(bool)
        output = ImageNormalization.apply(
            output,
            self.image_normalization,
            valid_mask=mask,
        ).astype(dtype, copy=False)

        if squeeze:
            return np.squeeze(output)
        return output

    def GetMeanSpectrum(self):
        total = np.zeros(self.depth, dtype=np.float64)
        count = 0

        for _, _, _, block in self._iter_blocks():
            spectra = self._process_spectra(
                block.reshape(-1, self.depth)
            )
            total += np.sum(spectra, axis=0, dtype=np.float64)
            count += spectra.shape[0]

        if count == 0:
            return total
        return total / count

    def GetMaxSpectrum(self):
        result = np.full(self.depth, -np.inf, dtype=np.float64)

        for _, _, _, block in self._iter_blocks():
            spectra = self._process_spectra(
                block.reshape(-1, self.depth)
            )
            result = np.maximum(
                result,
                np.max(spectra, axis=0),
            )

        return result

    def GetMaskArray(self):
        if self._valid_mask is not None:
            mask = self._valid_mask
        else:
            mask = np.ones(
                (self.depth_z, self.height, self.width),
                dtype=bool,
            )

        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        return mask.astype(np.ushort, copy=True)

    def GetMetaData(self) -> Dict[str, object]:
        metadata = dict(self.root.attrs)
        metadata.setdefault("Modality", self.GetModality())
        metadata.setdefault(
            "SpectralUnit", self.GetSpectralUnit()
        )
        metadata["pymirimaging.xs.n"] = int(self.depth)
        metadata["pymirimaging.xs.min"] = float(np.min(self.x_axis))
        metadata["pymirimaging.xs.max"] = float(np.max(self.x_axis))
        metadata["pymirimaging.xs"] = self.x_axis.tolist()
        metadata["pymirimaging.shape"] = self.GetShape().tolist()
        return metadata

    def GetParametersAsFormattedString(self):
        return (
            f"(normalization {self.normalization})\n"
            f"(spectral-selection {self.spectral_selection})\n"
            f"(tolerance {float(self.tolerance)})\n"
            f"(pooling {self.pooling})\n"
            f"(image-normalization {self.image_normalization})\n"
        )

    def _output_axis_for_requests(
        self, requested, selection=None, tolerance=None
    ):
        requested = np.asarray(requested, dtype=np.float64).reshape(-1)
        strategy = (
            self.spectral_selection
            if selection is None
            else value_of(selection)
        )
        tol = float(
            self.tolerance if tolerance is None else tolerance
        )

        output_axis = []
        groups = []
        for center in requested:
            indices = SpectralSelection.select_indices(
                self.x_axis,
                center,
                strategy=strategy,
                tolerance=tol,
            )
            values = self.x_axis[indices].astype(
                np.float64, copy=True
            )
            groups.append(values)
            output_axis.append(
                float(values[0])
                if values.size == 1
                else float(center)
            )
        return (
            np.asarray(output_axis, dtype=np.float64),
            groups,
            strategy,
            tol,
        )

    def WriteNRRD(
        self,
        output_path,
        wavenumbers=None,
        dtype=np.float32,
        selection=None,
        tolerance=None,
        use_compression=True,
        rotate_raw=True,
    ):
        """
        Export MIR data as an M2aia-compatible vector NRRD.

        wavenumbers=None exports every measured channel.
        For legacy/raw Zarr, a 90 degree counterclockwise rotation is
        applied by default, matching the supplied conversion script.
        Zarr created from NRRD is tagged orientation='m2aia_nrrd' and
        will not be rotated again.
        """
        if dtype not in (np.float32, np.float64):
            raise TypeError(
                "NRRD dtype must be np.float32 or np.float64."
            )

        if wavenumbers is None:
            requested = self.x_axis.copy()
            selection_for_write = "Exact"
        elif np.isscalar(wavenumbers):
            requested = np.asarray([wavenumbers], dtype=np.float64)
            selection_for_write = selection
        else:
            requested = np.asarray(
                wavenumbers, dtype=np.float64
            ).reshape(-1)
            selection_for_write = selection

        if requested.size == 0:
            raise ValueError("At least one wavenumber is required.")

        output_axis, groups, strategy, tol = (
            self._output_axis_for_requests(
                requested,
                selection=selection_for_write,
                tolerance=tolerance,
            )
        )

        channels = [
            self.GetArray(
                float(center),
                tol=tol,
                dtype=dtype,
                squeeze=False,
                selection=strategy,
            )
            for center in requested
        ]
        data = np.stack(channels, axis=-1)

        orientation = str(
            self.root.attrs.get("orientation", "raw_mir")
        )
        should_rotate = rotate_raw and orientation != "m2aia_nrrd"

        if should_rotate:
            data = np.rot90(data, axes=(1, 2))

        # data is (Z,Y,X,C). Write a 2D vector image when the source is
        # raw 2D and no pz/3D geometry is available.
        spacing = None
        origin = None
        direction = None

        if self._spacing is not None:
            spacing = self._spacing.copy()
            origin = self._origin.copy()
            direction = self._direction.copy()

            if should_rotate and len(spacing) >= 2:
                spacing = spacing.copy()
                spacing[0], spacing[1] = spacing[1], spacing[0]

        write_2d = (
            self.depth_z == 1
            and (
                spacing is None
                or len(spacing) == 2
            )
        )

        if write_2d:
            out_array = np.ascontiguousarray(data[0])
        else:
            out_array = np.ascontiguousarray(data)

        image = sitk.GetImageFromArray(
            out_array,
            isVector=True,
        )

        if spacing is not None:
            if write_2d:
                image.SetSpacing(tuple(spacing[:2]))
                image.SetOrigin(tuple(origin[:2]))
                if len(direction) == 4:
                    image.SetDirection(tuple(direction))
            else:
                if len(spacing) == 2:
                    raise ValueError(
                        "3D NRRD export requires Z spacing. "
                        "Pass pz in micrometres or store 3D spacing."
                    )
                image.SetSpacing(tuple(spacing[:3]))
                image.SetOrigin(tuple(origin[:3]))
                if len(direction) == 9:
                    image.SetDirection(tuple(direction))

        axis_text = format_wavenumbers(output_axis)
        image.SetMetaData("Type", axis_text)
        image.SetMetaData("Modality", "MIR")
        image.SetMetaData("SpectralUnit", "cm^-1")
        image.SetMetaData("m2aia.modality", "MIR")
        image.SetMetaData("m2aia.spectral_unit", "cm^-1")
        image.SetMetaData("m2aia.xaxis", axis_text)
        image.SetMetaData("m2aia_xaxis", axis_text)
        image.SetMetaData("wavenumbers", axis_text)
        image.SetMetaData("m2aia.xs", axis_text)
        image.SetMetaData("m2aia.xs.n", str(len(output_axis)))
        image.SetMetaData(
            "m2aia.xs.min", str(float(np.min(output_axis)))
        )
        image.SetMetaData(
            "m2aia.xs.max", str(float(np.max(output_axis)))
        )

        image.SetMetaData(
            "pymirimaging.processing.normalization",
            self.normalization,
        )
        image.SetMetaData(
            "pymirimaging.processing.spectral_selection",
            strategy,
        )
        image.SetMetaData(
            "pymirimaging.processing.tolerance_cm-1",
            f"{tol:.12g}",
        )
        image.SetMetaData(
            "pymirimaging.processing.pooling",
            self.pooling,
        )
        image.SetMetaData(
            "pymirimaging.processing.image_normalization",
            self.image_normalization,
        )

        source_metadata = self.root.attrs.get(
            "source_metadata", {}
        )
        if isinstance(source_metadata, dict):
            for key, value in source_metadata.items():
                upper = str(key).upper()
                if (
                    upper.startswith("NRRD_")
                    or upper.startswith("ITK_")
                ):
                    continue
                try:
                    image.SetMetaData(str(key), str(value))
                except RuntimeError:
                    pass

        # Reassert canonical spectral metadata after source metadata copy.
        image.SetMetaData("Type", axis_text)
        image.SetMetaData("Modality", "MIR")
        image.SetMetaData("SpectralUnit", "cm^-1")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            image,
            str(output_path),
            bool(use_compression),
        )
        return output_path

    def WriteZarr(
        self,
        output_path,
        wavenumbers=None,
        overwrite=False,
    ):
        """
        Write a Zarr dataset.

        If wavenumbers is None, the full cube is written and spectrum
        normalization is applied chunk-by-chunk. If selected wavenumbers
        are supplied, the configured selection/pooling/image normalization
        pipeline is used for those output channels.
        """
        output_path = Path(output_path)
        mode = "w" if overwrite else "w-"
        out = zarr.open(str(output_path), mode=mode)

        if wavenumbers is None:
            out_shape = (
                (self.height, self.width, self.depth)
                if self.data.ndim == 3
                else (
                    self.depth_z,
                    self.height,
                    self.width,
                    self.depth,
                )
            )
            chunks = getattr(self.data, "chunks", None)
            if chunks is None:
                chunks = out_shape

            cube_out = out.create_dataset(
                "hypercube",
                shape=out_shape,
                chunks=chunks,
                dtype=np.float32,
            )

            for z, y0, y1, block in self._iter_blocks():
                spectra = self._process_spectra(
                    block.reshape(-1, self.depth)
                ).reshape(block.shape)

                if self.data.ndim == 3:
                    cube_out[y0:y1, :, :] = spectra
                else:
                    cube_out[z, y0:y1, :, :] = spectra

            axis_out = self.x_axis.copy()

        else:
            requested = np.asarray(
                [wavenumbers]
                if np.isscalar(wavenumbers)
                else wavenumbers,
                dtype=np.float64,
            ).reshape(-1)

            images = []
            axis_out = []
            for center in requested:
                indices = self._select_spectral_indices(center)
                selected_values = self.x_axis[indices]
                axis_out.append(
                    float(selected_values[0])
                    if len(selected_values) == 1
                    else float(center)
                )
                images.append(
                    self.GetArray(
                        center,
                        squeeze=False,
                    )
                )

            stack = np.stack(images, axis=-1).astype(
                np.float32, copy=False
            )

            if self.data.ndim == 3:
                stack = stack[0]

            out.create_dataset(
                "hypercube",
                data=stack,
                chunks=True,
            )
            axis_out = np.asarray(axis_out, dtype=np.float64)

        out.create_dataset(
            "wvnm",
            data=np.asarray(axis_out, dtype=np.float64),
        )

        mask = self.GetMaskArray()
        if np.any(mask == 0):
            if self.data.ndim == 3:
                mask = mask[0]
            out.create_dataset(
                "mask",
                data=mask.astype(np.uint8),
                chunks=True,
            )

        for key, value in dict(self.root.attrs).items():
            out.attrs[str(key)] = json_safe(value)

        if self._spacing is not None:
            out.attrs["spacing"] = self._spacing.tolist()
            out.attrs["origin"] = self._origin.tolist()
            out.attrs["direction"] = self._direction.tolist()

        out.attrs["modality"] = self.GetModality()
        out.attrs["spectral_unit"] = self.GetSpectralUnit()
        out.attrs["spatial_unit"] = str(
            self.root.attrs.get("spatial_unit", "mm")
        )
        out.attrs["orientation"] = str(
            self.root.attrs.get("orientation", "raw_mir")
        )

        history = list(
            self.root.attrs.get("processing_history", [])
        )
        if self.normalization != "None":
            history.append(
                {
                    "operation": "normalization",
                    "method": self.normalization,
                }
            )
        out.attrs["processing_history"] = json_safe(history)

        return output_path
