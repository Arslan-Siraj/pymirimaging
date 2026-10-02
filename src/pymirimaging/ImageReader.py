from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

from ._helpers import json_safe


class ImageReader:
    """
    pyM2aia-style reader for ordinary scalar spatial images.

    Supported examples include TIFF, scalar NRRD, PNG, and JPEG.

    This class is intended for fluorescence/microscopy images, masks,
    and other non-spectral images. Pixel values and geometry are never
    changed automatically.
    """

    def __init__(self, image_path):
        self.image_path = str(image_path)
        self.image = sitk.ReadImage(self.image_path)
        self._refresh_array()

    def _refresh_array(self):
        self.data = sitk.GetArrayFromImage(self.image)

    def path(self):
        return Path(self.image_path)

    def name(self):
        return self.path().name

    def GetDimension(self):
        return int(self.image.GetDimension())

    def GetNumberOfComponentsPerPixel(self):
        return int(self.image.GetNumberOfComponentsPerPixel())

    def GetPixelType(self):
        return self.image.GetPixelIDTypeAsString()

    def GetArray(self, squeeze=False):
        arr = np.asarray(self.data)
        return np.squeeze(arr) if squeeze else arr

    def GetShape(self):
        return np.asarray(self.image.GetSize(), dtype=np.int32)

    def GetSpacing(self):
        return np.asarray(self.image.GetSpacing(), dtype=np.float64)

    def GetOrigin(self):
        return np.asarray(self.image.GetOrigin(), dtype=np.float64)

    def GetDirection(self):
        return np.asarray(self.image.GetDirection(), dtype=np.float64)

    def GetMetaData(self):
        return {
            key: self.image.GetMetaData(key)
            for key in self.image.GetMetaDataKeys()
        }

    def SetSpacing(self, *spacing):
        if len(spacing) == 1 and hasattr(spacing[0], "__iter__"):
            spacing = tuple(spacing[0])

        spacing = tuple(float(v) for v in spacing)

        if len(spacing) != self.GetDimension():
            raise ValueError(
                f"Expected {self.GetDimension()} spacing values, "
                f"got {len(spacing)}."
            )

        if not all(np.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("All spacing values must be positive and finite.")

        self.image.SetSpacing(spacing)
        return self

    def SetOrigin(self, *origin):
        if len(origin) == 1 and hasattr(origin[0], "__iter__"):
            origin = tuple(origin[0])

        origin = tuple(float(v) for v in origin)

        if len(origin) != self.GetDimension():
            raise ValueError(
                f"Expected {self.GetDimension()} origin values, "
                f"got {len(origin)}."
            )

        if not all(np.isfinite(v) for v in origin):
            raise ValueError("All origin values must be finite.")

        self.image.SetOrigin(origin)
        return self

    def SetDirection(self, direction):
        if isinstance(direction, str):
            if direction.lower() != "identity":
                raise ValueError(
                    "String direction must be 'identity', or provide "
                    "the flattened numeric direction matrix."
                )
            direction = np.eye(self.GetDimension(), dtype=float).ravel()

        direction = tuple(float(v) for v in direction)
        dim = self.GetDimension()

        if len(direction) != dim * dim:
            raise ValueError(
                f"Expected {dim * dim} direction values for a "
                f"{dim}D image, got {len(direction)}."
            )

        if not all(np.isfinite(v) for v in direction):
            raise ValueError("All direction values must be finite.")

        self.image.SetDirection(direction)
        return self

    def SetGeometry(
        self,
        spacing=None,
        origin=None,
        direction=None,
    ):
        """
        Explicitly set physical image geometry.

        Only values supplied by the user are changed. No physical
        geometry is guessed from pixel dimensions.
        """
        if spacing is not None:
            self.SetSpacing(spacing)

        if origin is not None:
            self.SetOrigin(origin)

        if direction is not None:
            self.SetDirection(direction)

        return self

    def SetMetaData(self, key, value):
        self.image.SetMetaData(str(key), str(value))
        return self

    def SelectChannel(self, channel):
        """
        Convert a vector image to a scalar image by selecting one channel.

        Parameters
        ----------
        channel : int
            Zero-based channel index.

            For a standard RGB image:
            0 = red
            1 = green
            2 = blue
        """
        channel = int(channel)

        number_of_components = (
            self.GetNumberOfComponentsPerPixel()
        )

        if number_of_components == 1:
            if channel != 0:
                raise ValueError(
                    "Image is already scalar. "
                    "Only channel=0 is valid."
                )
            return self

        if (
            channel < 0
            or channel >= number_of_components
        ):
            raise ValueError(
                f"Invalid channel {channel}. "
                f"Image has {number_of_components} components."
            )

        metadata = self.GetMetaData()

        scalar = sitk.VectorIndexSelectionCast(
            self.image,
            channel,
        )

        for key, value in metadata.items():
            try:
                scalar.SetMetaData(
                    str(key),
                    str(value),
                )
            except RuntimeError:
                pass

        scalar.SetMetaData(
            "pymirimaging.image.selected_channel",
            str(channel),
        )

        self.image = scalar
        self._refresh_array()

        return self

    def PromoteTo3D(
        self,
        z_spacing=0.01,
        z_origin=0.0,
    ):
        """
        Promote a 2D scalar image to one-slice 3D.

        Current X/Y spacing and origin are preserved. Z spacing and
        origin are explicitly provided by the user.
        """
        if self.GetDimension() == 3:
            return self

        if self.GetDimension() != 2:
            raise ValueError(
                "PromoteTo3D() currently supports only 2D images."
            )

        if self.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(
                "PromoteTo3D() currently supports only scalar images."
            )

        z_spacing = float(z_spacing)
        z_origin = float(z_origin)

        if not np.isfinite(z_spacing) or z_spacing <= 0:
            raise ValueError("z_spacing must be positive and finite.")

        old_spacing = self.image.GetSpacing()
        old_origin = self.image.GetOrigin()
        metadata = self.GetMetaData()

        image3d = sitk.JoinSeries(self.image)

        image3d.SetSpacing(
            (
                float(old_spacing[0]),
                float(old_spacing[1]),
                z_spacing,
            )
        )
        image3d.SetOrigin(
            (
                float(old_origin[0]),
                float(old_origin[1]),
                z_origin,
            )
        )
        image3d.SetDirection(
            (
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            )
        )

        for key, value in metadata.items():
            try:
                image3d.SetMetaData(str(key), str(value))
            except RuntimeError:
                pass

        self.image = image3d
        self._refresh_array()
        return self

    def SetPixelCenterOrigin(self):
        """
        Set X/Y origin to half of X/Y spacing.

        Use this only when the acquisition/registration convention
        requires the first pixel center to define the physical origin.
        """
        if self.GetDimension() < 2:
            raise ValueError(
                "SetPixelCenterOrigin() requires at least a 2D image."
            )

        spacing = self.GetSpacing()
        origin = self.GetOrigin().copy()

        origin[0] = spacing[0] / 2.0
        origin[1] = spacing[1] / 2.0

        self.image.SetOrigin(tuple(float(v) for v in origin))
        return self

    def WriteNRRD(
        self,
        output_path,
        use_compression=True,
    ):
        """
        Write the current image to NRRD without changing its pixel
        values, spacing, origin, direction, or orientation.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.image.SetMetaData(
            "pymirimaging.source_format",
            self.path().suffix.lower(),
        )

        sitk.WriteImage(
            self.image,
            str(output_path),
            bool(use_compression),
        )

        return output_path

    def WriteZarr(
        self,
        output_path,
        overwrite=False,
    ):
        """
        Write the current scalar image to Zarr dataset ``image``.
        """
        output_path = Path(output_path)

        out = zarr.open(
            str(output_path),
            mode="w" if overwrite else "w-",
        )

        out.create_dataset(
            "image",
            data=np.asarray(self.data),
            chunks=True,
        )

        out.attrs["spacing"] = self.GetSpacing().tolist()
        out.attrs["origin"] = self.GetOrigin().tolist()
        out.attrs["direction"] = self.GetDirection().tolist()
        out.attrs["source_format"] = self.path().suffix.lower()
        out.attrs["source_metadata"] = json_safe(self.GetMetaData())

        return output_path
