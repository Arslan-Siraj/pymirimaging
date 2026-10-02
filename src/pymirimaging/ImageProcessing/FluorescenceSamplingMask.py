from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from sklearn.neighbors import NearestNeighbors
from skimage import filters, morphology, measure, util
from skimage.draw import disk
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_erosion,
    disk as morph_disk,
)


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


class FluorescenceSamplingMask:
    """
    Fluorescence sampling mask reproducing the reference masking script.

    The selection algorithm intentionally follows the supplied script,
    including:

    - Gaussian sigma = 4
    - tissue threshold = 3 after img_as_ubyte
    - tissue minimum size = 5000 pixels
    - closing disk radius = 30 pixels
    - hole area threshold = 5000 pixels
    - largest connected tissue component only
    - 40 erosions with disk radius = 2 pixels
    - fluorescence threshold = 20
    - area criterion 2..15 pixels OR roundness >= 0.75
    - pixel_size_um = 3 * 1.621
    - nearest-neighbour filtering with sklearn NearestNeighbors
    - nearest-neighbour range = 140..500 um
    - random seed = 42
    - at most 100 objects
    - circle radius = 40 um

    By default, output construction and geometry follow the supplied script:
    spacing = (source_sx, source_sy, 0.01), origin = (0, 0, 0),
    identity direction, and shrink factors = [1, 1, 1].

    Set preserve_source_geometry=True to copy spacing, origin, and direction
    from the imported 3D NRRD while leaving the mask-selection algorithm
    unchanged.

    Using the defaults reproduces the supplied script's masking criteria.
    """

    def __init__(self, source):
        self.source = _as_sitk_image(source)

        if self.source.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(
                "FluorescenceSamplingMask requires a scalar image."
            )

        # Match the reference script exactly.
        img = sitk.GetArrayFromImage(self.source)
        img = np.squeeze(img)

        if img.ndim != 2:
            raise ValueError(
                "The reference masking workflow requires a 2D image "
                "or a single-slice 3D scalar image."
            )

        self.image_array = img

        self.tissue_mask = None
        self.tissue_mask_eroded = None
        self.filtered_object_mask = None
        self.centroids_rc = np.empty((0, 2), dtype=float)
        self.final_mask = None
        self.binary_mask = None
        self.mask_image = None
        self.parameters = {}

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
        pixel_size_um=3 * 1.621,
        min_nn_um=140,
        max_nn_um=500,
        max_objects=100,
        circle_radius_um=40,
        random_seed=42,
        preserve_source_geometry=False,
    ):
        """
        Generate the mask using the supplied reference-script algorithm.

        For an exact reproduction, call Generate() with the defaults.
        """
        img = self.image_array

        # --------------------------------------------------------------
        # Smooth
        # Reference:
        # blur = filters.gaussian(img, sigma=4)
        # blur_uint8 = util.img_as_ubyte(blur)
        # --------------------------------------------------------------
        blur = filters.gaussian(
            img,
            sigma=tissue_gaussian_sigma,
        )
        blur_uint8 = util.img_as_ubyte(blur)

        # --------------------------------------------------------------
        # Tissue mask
        # Reference:
        # thresh = 3
        # mask = blur_uint8 > thresh
        # --------------------------------------------------------------
        thresh = tissue_threshold
        mask = blur_uint8 > thresh

        # Remove tiny regions.
        mask = morphology.remove_small_objects(
            mask,
            min_size=tissue_min_size,
        )

        # Close small gaps.
        mask = binary_closing(
            mask,
            footprint=morph_disk(tissue_closing_radius),
        )

        # Fill holes.
        mask = ndi.binary_fill_holes(mask)

        # Remove small holes.
        mask = morphology.remove_small_holes(
            mask,
            area_threshold=tissue_hole_area,
        )

        # Keep largest connected component.
        labels = measure.label(mask)
        regions = measure.regionprops(labels)

        # Intentionally match the reference script:
        # max() raises if there are no tissue regions.
        largest = max(regions, key=lambda r: r.area)
        tissue_mask = labels == largest.label

        # Erode tissue exactly as in the reference script.
        tissue_mask_eroded = tissue_mask.copy()

        for _ in range(tissue_erosion_iterations):
            tissue_mask_eroded = binary_erosion(
                tissue_mask_eroded,
                footprint=morph_disk(tissue_erosion_radius),
            )

        # --------------------------------------------------------------
        # Fluorescence threshold inside eroded tissue
        # --------------------------------------------------------------
        threshold = object_threshold
        th = threshold

        mask = img > th
        mask[tissue_mask_eroded == 0] = 0

        labels = label(mask)

        filtered_mask = np.zeros_like(
            mask,
            dtype=bool,
        )

        # --------------------------------------------------------------
        # Area / circularity filtering
        # IMPORTANT: preserve OR exactly from reference script.
        # --------------------------------------------------------------
        min_size = object_min_size
        max_size = object_max_size

        for region in regionprops(labels):
            area = region.area

            if region.perimeter > 0:
                circularity = (
                    4 * np.pi * area
                    / (region.perimeter ** 2)
                )
            else:
                circularity = 0

            if (
                min_size <= area <= max_size
                or circularity >= min_roundness
            ):
                filtered_mask[
                    labels == region.label
                ] = True

        # --------------------------------------------------------------
        # Physical constants / pixel-space thresholds
        # Match reference script exactly.
        # --------------------------------------------------------------
        pixel_size_um = float(pixel_size_um)
        circle_radius_um = float(circle_radius_um)

        circle_radius_px = int(
            round(
                circle_radius_um
                / pixel_size_um
            )
        )

        min_nn_px = (
            float(min_nn_um)
            / pixel_size_um
        )
        max_nn_px = (
            float(max_nn_um)
            / pixel_size_um
        )

        # --------------------------------------------------------------
        # Label + centroids
        # --------------------------------------------------------------
        labels = label(filtered_mask > 0)

        props = regionprops(labels)
        centroids = np.array(
            [p.centroid for p in props]
        )

        # --------------------------------------------------------------
        # Nearest-neighbour filtering
        # Match reference script exactly:
        # sklearn.neighbors.NearestNeighbors(n_neighbors=2)
        # --------------------------------------------------------------
        if len(centroids) >= 1:
            nbrs = NearestNeighbors(
                n_neighbors=2
            )
            nbrs.fit(centroids)

            distances, _ = nbrs.kneighbors(
                centroids
            )

            # Remove self-distance.
            nn = distances[:, 1:]

            # The reference comment says "five", but its code uses
            # only the one non-self neighbour. Preserve the code.
            mean_nn = nn.mean(axis=1)

            keep = (
                (mean_nn >= min_nn_px)
                & (mean_nn <= max_nn_px)
            )

            centroids = centroids[keep]

        # --------------------------------------------------------------
        # Random subset
        # --------------------------------------------------------------
        rng = np.random.default_rng(
            random_seed
        )

        if len(centroids) > max_objects:
            idx = rng.choice(
                len(centroids),
                max_objects,
                replace=False,
            )
            centroids = centroids[idx]

        # --------------------------------------------------------------
        # Create final circular ROI mask
        # --------------------------------------------------------------
        final_mask = np.zeros(
            labels.shape,
            dtype=bool,
        )

        for r, c in centroids:
            rr, cc = disk(
                (r, c),
                circle_radius_px,
                shape=final_mask.shape,
            )

            final_mask[rr, cc] = True

        # --------------------------------------------------------------
        # Reproduce the reference mask conversion path.
        # Plotting itself is omitted because it does not change the result.
        # --------------------------------------------------------------
        final_mask_for_output = final_mask.astype(
            np.float32
        )
        final_mask_for_output[
            final_mask_for_output == 0
        ] = np.nan

        if np.any(
            np.isfinite(final_mask_for_output)
        ):
            final_mask_for_output = (
                final_mask_for_output
                / np.nanmax(final_mask_for_output)
            )

        binary_mask = (
            np.nan_to_num(
                final_mask_for_output
            )
            > 0
        )

        binary_mask = (
            binary_mask.astype(np.uint8)
            * 255
        )

        # --------------------------------------------------------------
        # Reproduce original RGB -> grayscale output construction.
        # --------------------------------------------------------------
        rgb = np.stack(
            [binary_mask] * 3,
            axis=-1,
        )

        out_arr = rgb[np.newaxis, ...]

        out_img = sitk.GetImageFromArray(
            out_arr,
            isVector=True,
        )

        r = sitk.VectorIndexSelectionCast(
            out_img,
            0,
            sitk.sitkFloat32,
        )
        g = sitk.VectorIndexSelectionCast(
            out_img,
            1,
            sitk.sitkFloat32,
        )
        b = sitk.VectorIndexSelectionCast(
            out_img,
            2,
            sitk.sitkFloat32,
        )

        gray = (
            0.299 * r
            + 0.587 * g
            + 0.114 * b
        )

        out_img = sitk.Cast(
            gray,
            sitk.sitkUInt8,
        )

        # --------------------------------------------------------------
        # Output geometry.
        #
        # False = reproduce the supplied reference script exactly.
        # True  = preserve the geometry of the imported NRRD.
        #
        # This option changes ONLY physical geometry. It does not change
        # tissue detection, object selection, nearest-neighbour filtering,
        # random selection, or final mask pixels.
        # --------------------------------------------------------------
        if preserve_source_geometry:
            if self.source.GetDimension() != 3:
                raise ValueError(
                    "preserve_source_geometry=True requires a 3D "
                    "single-slice source image."
                )

            if tuple(out_img.GetSize()) != tuple(self.source.GetSize()):
                raise ValueError(
                    "Cannot copy source geometry because source and "
                    "mask sizes differ."
                )

            out_img.SetSpacing(
                self.source.GetSpacing()
            )
            out_img.SetOrigin(
                self.source.GetOrigin()
            )
            out_img.SetDirection(
                self.source.GetDirection()
            )

        else:
            # Match the supplied reference script exactly.
            spacing = self.source.GetSpacing()
            sx = spacing[0]
            sy = spacing[1]

            out_img.SetSpacing(
                (sx, sy, 0.01)
            )

            out_img.SetOrigin(
                (0.0, 0.0, 0.0)
            )

            out_img.SetDirection(
                (
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                )
            )

        # --------------------------------------------------------------
        # Optional downsampling from reference script:
        # x_filter = 1
        # y_filter = 1
        # --------------------------------------------------------------
        x_filter = 1
        y_filter = 1

        shrink = sitk.ShrinkImageFilter()
        shrink.SetShrinkFactors(
            [x_filter, y_filter, 1]
        )

        out_img = shrink.Execute(out_img)

        # Store state for inspection.
        self.tissue_mask = tissue_mask
        self.tissue_mask_eroded = tissue_mask_eroded
        self.filtered_object_mask = filtered_mask
        self.centroids_rc = centroids
        self.final_mask = final_mask
        self.binary_mask = binary_mask
        self.mask_image = out_img

        self.parameters = {
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
            "pixel_size_um": pixel_size_um,
            "min_nn_um": min_nn_um,
            "max_nn_um": max_nn_um,
            "max_objects": max_objects,
            "circle_radius_um": circle_radius_um,
            "random_seed": random_seed,
            "preserve_source_geometry": preserve_source_geometry,
            "selected_objects": len(centroids),
        }

        return self

    def GetMaskImage(self):
        if self.mask_image is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.mask_image

    def GetArray(self, squeeze=True):
        arr = sitk.GetArrayFromImage(
            self.GetMaskImage()
        )
        return (
            np.squeeze(arr)
            if squeeze
            else arr
        )

    def GetTissueMask(self):
        if self.tissue_mask is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.tissue_mask.copy()

    def GetErodedTissueMask(self):
        if self.tissue_mask_eroded is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.tissue_mask_eroded.copy()

    def GetFilteredObjectMask(self):
        if self.filtered_object_mask is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.filtered_object_mask.copy()

    def GetCentroids(self):
        return self.centroids_rc.copy()

    def GetParameters(self):
        return dict(self.parameters)

    def WriteNRRD(
        self,
        output_path,
        use_compression=False,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sitk.WriteImage(
            self.GetMaskImage(),
            str(output_path),
            bool(use_compression),
        )

        return output_path
