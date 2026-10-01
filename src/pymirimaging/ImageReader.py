from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

from ._helpers import json_safe


class ImageReader:
    """
    Simple pyM2aia-style reader for ordinary spatial images such as
    scalar NRRD, TIFF, PNG, or JPEG.
    """

    def __init__(self, image_path):
        self.image_path = str(image_path)
        self.image = sitk.ReadImage(self.image_path)
        self.data = sitk.GetArrayFromImage(self.image)

    def path(self):
        return Path(self.image_path)

    def GetArray(self, squeeze=False):
        result = np.asarray(self.data)
        return np.squeeze(result) if squeeze else result

    def GetShape(self):
        return np.asarray(self.image.GetSize(), dtype=np.int32)

    def GetSpacing(self):
        return np.asarray(
            self.image.GetSpacing(), dtype=np.float64
        )

    def GetOrigin(self):
        return np.asarray(
            self.image.GetOrigin(), dtype=np.float64
        )

    def GetDirection(self):
        return np.asarray(
            self.image.GetDirection(), dtype=np.float64
        )

    def GetMetaData(self):
        return {
            key: self.image.GetMetaData(key)
            for key in self.image.GetMetaDataKeys()
        }

    def WriteNRRD(self, output_path, use_compression=True):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            self.image,
            str(output_path),
            bool(use_compression),
        )
        return output_path

    def WriteZarr(self, output_path, overwrite=False):
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
        out.attrs["source_metadata"] = json_safe(
            self.GetMetaData()
        )
        return output_path
