from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure, morphology, util
from skimage.draw import disk


def _as_sitk_image(source):
    if isinstance(source, sitk.Image):
        return source

    if hasattr(source, "image") and isinstance(source.image, sitk.Image):
        return source.image

    if isinstance(source, (str, Path)):
        return sitk.ReadImage(str(source))

    raise TypeError(
        "source must be a pymirimaging.ImageReader, SimpleITK.Image, "
        "or image filename."
    )


def _copy_information_and_metadata(source, target):
    target.CopyInformation(source)

    for key in source.GetMetaDataKeys():
        try:
            target.SetMetaData(key, source.GetMetaData(key))
        except RuntimeError:
            pass


class FluorescenceSamplingMask:
    """
    Generate a fluorescence-derived spatial sampling mask.

    This class packages the current fluorescence masking workflow:

    1. Gaussian smoothing for broad tissue detection.
    2. Explicit tissue threshold.
    3. Tissue cleanup and largest-component selection.
    4. Tissue erosion to avoid boundary regions.
    5. Explicit object-intensity threshold.
    6. Candidate filtering by area/circularity.
    7. Nearest-neighbour filtering in physical units (micrometres).
    8. Reproducible optional random subsampling.
    9. Circular ROI creation around retained centroids.

    The generated mask is a scalar UInt8 image with values 0 and 255.
    Physical geometry and source metadata are copied from the input.

    No automatic Otsu thresholding is used.
    """

    def __init__(self, source):
        self.source = _as_sitk_image(source)

        if self.source.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(
                "FluorescenceSamplingMask requires a scalar image."
            )

        data = sitk.GetArrayFromImage(self.source)

        if data.ndim == 3:
            if data.shape[0] != 1:
                raise ValueError(
                    "Only 2D or single-slice 3D images are currently "
                    "supported."
                )
            data = data[0]
        elif data.ndim != 2:
            raise ValueError(
                "Only 2D or single-slice 3D images are currently supported."
            )

        self.image_array = np.asarray(data)

        self.tissue_mask = None
        self.tissue_mask_eroded = None
        self.filtered_object_mask = None
        self.centroids_rc = np.empty((0, 2), dtype=float)
        self.final_mask = None
        self.mask_image = None
        self.parameters = {}

    def _spacing_um(self):
        spacing = self.source.GetSpacing()

        sx_um = float(spacing[0]) * 1000.0
        sy_um = float(spacing[1]) * 1000.0

        if (
            not np.isfinite(sx_um)
            or not np.isfinite(sy_um)
            or sx_um <= 0
            or sy_um <= 0
        ):
            raise ValueError(
                "Source image must have positive finite X/Y spacing."
            )

        return sx_um, sy_um

    def Generate(
        self,
        object_threshold=20,
        object_min_size=2,
        object_max_size=15,
        min_roundness=0.75,
        tissue_gaussian_sigma=4,
        tissue_threshold=3,
        tissue_min_size=5000,
        tissue_closing_radius=30,
        tissue_hole_area=5000,
        tissue_erosion_iterations=40,
        tissue_erosion_radius=2,
        min_nn_um=140,
        max_nn_um=500,
        max_objects=100,
        circle_radius_um=40,
        random_seed=42,
        foreground_value=255,
    ):
        """
        Generate the sampling mask.

        The defaults reproduce the parameter values from the current
        fluorescence-mask script, while leaving them user-configurable.
        """
        img = self.image_array
        sx_um, sy_um = self._spacing_um()

        # 1. Broad tissue mask.
        blur = filters.gaussian(
            img,
            sigma=float(tissue_gaussian_sigma),
        )
        blur_uint8 = util.img_as_ubyte(blur)

        tissue_mask = blur_uint8 > float(tissue_threshold)

        tissue_mask = morphology.remove_small_objects(
            tissue_mask,
            min_size=int(tissue_min_size),
        )

        tissue_mask = morphology.binary_closing(
            tissue_mask,
            footprint=morphology.disk(int(tissue_closing_radius)),
        )

        tissue_mask = ndi.binary_fill_holes(tissue_mask)

        tissue_mask = morphology.remove_small_holes(
            tissue_mask,
            area_threshold=int(tissue_hole_area),
        )

        tissue_labels = measure.label(tissue_mask)
        tissue_regions = measure.regionprops(tissue_labels)

        if not tissue_regions:
            raise ValueError(
                "No tissue component found. Adjust tissue_threshold "
                "or tissue_min_size."
            )

        largest = max(
            tissue_regions,
            key=lambda region: region.area,
        )

        tissue_mask = tissue_labels == largest.label
        tissue_mask_eroded = tissue_mask.copy()

        for _ in range(int(tissue_erosion_iterations)):
            tissue_mask_eroded = morphology.binary_erosion(
                tissue_mask_eroded,
                footprint=morphology.disk(int(tissue_erosion_radius)),
            )

        # 2. Candidate objects inside eroded tissue.
        candidate_mask = img > float(object_threshold)
        candidate_mask[tissue_mask_eroded == 0] = False

        object_labels = measure.label(candidate_mask)

        filtered_mask = np.zeros_like(
            candidate_mask,
            dtype=bool,
        )

        for region in measure.regionprops(object_labels):
            area = float(region.area)

            if region.perimeter > 0:
                circularity = (
                    4.0
                    * np.pi
                    * area
                    / (float(region.perimeter) ** 2)
                )
            else:
                circularity = 0.0

            # Preserve the rule from the existing script.
            keep = (
                (
                    float(object_min_size)
                    <= area
                    <= float(object_max_size)
                )
                or circularity >= float(min_roundness)
            )

            if keep:
                filtered_mask[
                    object_labels == region.label
                ] = True

        # 3. Object centroids.
        filtered_labels = measure.label(filtered_mask)
        props = measure.regionprops(filtered_labels)

        centroids_rc = np.asarray(
            [region.centroid for region in props],
            dtype=float,
        )

        if centroids_rc.size == 0:
            centroids_rc = np.empty((0, 2), dtype=float)

        # 4. Nearest-neighbour filtering in micrometres.
        if len(centroids_rc) >= 2:
            centroids_um = np.column_stack(
                (
                    centroids_rc[:, 0] * sy_um,
                    centroids_rc[:, 1] * sx_um,
                )
            )

            tree = cKDTree(centroids_um)
            distances, _ = tree.query(centroids_um, k=2)
            nearest_um = distances[:, 1]

            keep = (
                (nearest_um >= float(min_nn_um))
                & (nearest_um <= float(max_nn_um))
            )

            centroids_rc = centroids_rc[keep]

        elif len(centroids_rc) == 1:
            # Nearest-neighbour constraints cannot be evaluated
            # for a single detected object.
            centroids_rc = np.empty((0, 2), dtype=float)

        # 5. Reproducible subset.
        max_objects = int(max_objects)

        if max_objects < 0:
            raise ValueError("max_objects must be >= 0.")

        if len(centroids_rc) > max_objects:
            rng = np.random.default_rng(int(random_seed))
            idx = rng.choice(
                len(centroids_rc),
                max_objects,
                replace=False,
            )
            centroids_rc = centroids_rc[idx]

        # 6. Circular sampling regions.
        # The current acquisition is isotropic in X/Y. The mean spacing
        # keeps this usable if tiny numerical differences are present.
        pixel_size_um = (sx_um + sy_um) / 2.0

        circle_radius_px = int(
            round(float(circle_radius_um) / pixel_size_um)
        )

        if circle_radius_px < 1:
            raise ValueError(
                "circle_radius_um is smaller than one image pixel."
            )

        final_mask = np.zeros(
            filtered_labels.shape,
            dtype=bool,
        )

        for row, col in centroids_rc:
            rr, cc = disk(
                (row, col),
                circle_radius_px,
                shape=final_mask.shape,
            )
            final_mask[rr, cc] = True

        output_array = (
            final_mask.astype(np.uint8)
            * int(foreground_value)
        )

        if self.source.GetDimension() == 3:
            output_array = output_array[np.newaxis, ...]

        mask_image = sitk.GetImageFromArray(
            output_array,
            isVector=False,
        )

        _copy_information_and_metadata(
            self.source,
            mask_image,
        )

        parameters = {
            "object_threshold": object_threshold,
            "object_min_size": object_min_size,
            "object_max_size": object_max_size,
            "min_roundness": min_roundness,
            "tissue_gaussian_sigma": tissue_gaussian_sigma,
            "tissue_threshold": tissue_threshold,
            "tissue_min_size": tissue_min_size,
            "tissue_closing_radius": tissue_closing_radius,
            "tissue_hole_area": tissue_hole_area,
            "tissue_erosion_iterations": tissue_erosion_iterations,
            "tissue_erosion_radius": tissue_erosion_radius,
            "min_nn_um": min_nn_um,
            "max_nn_um": max_nn_um,
            "max_objects": max_objects,
            "circle_radius_um": circle_radius_um,
            "random_seed": random_seed,
            "foreground_value": foreground_value,
            "pixel_size_x_um": sx_um,
            "pixel_size_y_um": sy_um,
            "selected_objects": len(centroids_rc),
        }

        mask_image.SetMetaData(
            "pymirimaging.modality",
            "mask",
        )
        mask_image.SetMetaData(
            "pymirimaging.mask.type",
            "fluorescence_sampling_mask",
        )
        mask_image.SetMetaData(
            "pymirimaging.mask.method",
            "fluorescence_sampling",
        )

        for key, value in parameters.items():
            mask_image.SetMetaData(
                f"pymirimaging.mask.{key}",
                str(value),
            )

        self.tissue_mask = tissue_mask
        self.tissue_mask_eroded = tissue_mask_eroded
        self.filtered_object_mask = filtered_mask
        self.centroids_rc = centroids_rc
        self.final_mask = final_mask
        self.mask_image = mask_image
        self.parameters = parameters

        return self

    def GetMaskImage(self):
        if self.mask_image is None:
            raise RuntimeError("Generate() must be called first.")
        return self.mask_image

    def GetArray(self, squeeze=True):
        arr = sitk.GetArrayFromImage(self.GetMaskImage())
        return np.squeeze(arr) if squeeze else arr

    def GetTissueMask(self):
        if self.tissue_mask is None:
            raise RuntimeError("Generate() must be called first.")
        return self.tissue_mask.copy()

    def GetErodedTissueMask(self):
        if self.tissue_mask_eroded is None:
            raise RuntimeError("Generate() must be called first.")
        return self.tissue_mask_eroded.copy()

    def GetFilteredObjectMask(self):
        if self.filtered_object_mask is None:
            raise RuntimeError("Generate() must be called first.")
        return self.filtered_object_mask.copy()

    def GetCentroids(self):
        return self.centroids_rc.copy()

    def GetParameters(self):
        return dict(self.parameters)

    def WriteNRRD(
        self,
        output_path,
        use_compression=True,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(
            self.GetMaskImage(),
            str(output_path),
            bool(use_compression),
        )

        return output_path
