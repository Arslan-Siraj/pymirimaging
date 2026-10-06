from pathlib import Path
import json
import re
from typing import Optional, Union

import matplotlib.pyplot as plt
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


PathLike = Union[str, Path]


def _as_sitk_image(source):
    if isinstance(source, sitk.Image):
        return sitk.Image(source)

    if hasattr(source, "image") and isinstance(source.image, sitk.Image):
        return sitk.Image(source.image)

    if isinstance(source, (str, Path)):
        return sitk.ReadImage(str(source))

    raise TypeError(
        "source must be a pymirimaging.ImageReader, SimpleITK.Image, "
        "or image filename."
    )


def _display_normalize(arr, low=1.0, high=99.5):
    """Robust [0, 1] normalization used only for QC figures."""
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr)

    out = np.zeros_like(arr, dtype=np.float64)

    if not np.any(valid):
        return out

    lo, hi = np.percentile(arr[valid], [low, high])

    if hi <= lo:
        return out

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return out


class FluorescenceSamplingMask:
    """
    Fluorescence sampling-mask generator with optional step-by-step QC output.

    The default plaque-selection behavior remains compatible with the original
    reference workflow:

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
    - nearest-neighbour range = 140..500 um
    - random seed = 42
    - at most 100 objects
    - circle radius = 40 um

    New QC options do not change plaque-selection behavior unless the caller
    explicitly changes one of the selection parameters.

    Typical diagnostic call
    -----------------------
    masker = FluorescenceSamplingMask(if_path)

    masker.Generate(
        output_dir="S1_N3_plaque_mask_output",
        sample_name="S1_N3",
        save_figures=True,
        save_summary=True,
        show_figures=False,
        preserve_source_geometry=True,
    )

    masker.WriteNRRD(
        "S1_N3_plaque_mask_output/S1_N3_IF_plaque_sampling_mask.nrrd"
    )
    """

    def __init__(self, source):
        self.source = _as_sitk_image(source)

        if self.source.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(
                "FluorescenceSamplingMask requires a scalar image."
            )

        img = sitk.GetArrayFromImage(self.source)
        img = np.squeeze(img)

        if img.ndim != 2:
            raise ValueError(
                "The masking workflow requires a 2D image "
                "or a single-slice 3D scalar image."
            )

        self.image_array = img

        # Main outputs / original public state.
        self.tissue_mask = None
        self.tissue_mask_eroded = None
        self.filtered_object_mask = None
        self.centroids_rc = np.empty((0, 2), dtype=float)
        self.final_mask = None
        self.binary_mask = None
        self.mask_image = None
        self.parameters = {}

        # Additional QC state.
        self.blur = None
        self.blur_uint8 = None
        self.tissue_threshold_mask = None
        self.tissue_cleaned_mask = None
        self.object_candidate_mask = None
        self.centroids_before_nn = np.empty((0, 2), dtype=float)
        self.centroids_after_nn = np.empty((0, 2), dtype=float)
        self.selected_centroids_rc = np.empty((0, 2), dtype=float)

        self.stage_counts = {}
        self.summary = None
        self.saved_figures = []
        self._figure_counter = 0

        self.output_dir = None
        self.figures_dir = None
        self.summary_json_path = None
        self.sample_name = "sample"
        self.save_figures = False
        self.show_figures = False
        self.save_summary = False

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
        use_nearest_neighbor_filter=True,
        output_dir: Optional[PathLike] = None,
        sample_name="sample",
        save_figures=False,
        show_figures=False,
        save_summary=False,
    ):
        """
        Generate the sampling mask.

        The default algorithm reproduces the previous reference workflow.
        The additional output/QC parameters only control reporting.

        Parameters added for QC
        -----------------------
        use_nearest_neighbor_filter : bool, default True
            Keep the original 140..500 um nearest-neighbour filtering.
            Set False only when you intentionally want to inspect/retain
            candidates without that spacing filter.

        output_dir : path or None
            Directory for optional figures and summary JSON.

        sample_name : str
            Prefix used in the summary filename and figure titles.

        save_figures : bool
            Save intermediate PNG figures in output_dir / "figures".

        show_figures : bool
            Show figures interactively.

        save_summary : bool
            Save a JSON summary with the number of objects remaining after
            each plaque-selection stage.

        Notes
        -----
        ``max_objects=None`` is accepted and means "do not randomly cap the
        number of retained objects".  The historical default remains 100.
        """
        img = self.image_array

        self.sample_name = str(sample_name)
        self.save_figures = bool(save_figures)
        self.show_figures = bool(show_figures)
        self.save_summary = bool(save_summary)

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.figures_dir = self.output_dir / "figures"
            self.summary_json_path = (
                self.output_dir
                / f"{self.sample_name}_fluorescence_plaque_mask_summary.json"
            )

            if self.save_figures:
                self.figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            self.figures_dir = None
            self.summary_json_path = None

        if (self.save_figures or self.save_summary) and self.output_dir is None:
            raise ValueError(
                "output_dir is required when save_figures=True or "
                "save_summary=True."
            )

        if float(pixel_size_um) <= 0:
            raise ValueError("pixel_size_um must be > 0.")

        if float(circle_radius_um) <= 0:
            raise ValueError("circle_radius_um must be > 0.")

        if max_objects is not None and int(max_objects) < 1:
            raise ValueError("max_objects must be >= 1 or None.")

        # --------------------------------------------------------------
        # STEP 1 — smooth IF
        # --------------------------------------------------------------
        blur = filters.gaussian(
            img,
            sigma=tissue_gaussian_sigma,
        )
        blur_uint8 = util.img_as_ubyte(blur)

        self.blur = blur
        self.blur_uint8 = blur_uint8

        # --------------------------------------------------------------
        # STEP 2 — initial tissue threshold
        # --------------------------------------------------------------
        tissue_threshold_mask = (
            blur_uint8 > tissue_threshold
        )

        self.tissue_threshold_mask = tissue_threshold_mask.copy()

        # --------------------------------------------------------------
        # STEP 3 — clean tissue mask
        # --------------------------------------------------------------
        cleaned = morphology.remove_small_objects(
            tissue_threshold_mask,
            min_size=tissue_min_size,
        )

        cleaned = binary_closing(
            cleaned,
            footprint=morph_disk(tissue_closing_radius),
        )

        cleaned = ndi.binary_fill_holes(cleaned)

        cleaned = morphology.remove_small_holes(
            cleaned,
            area_threshold=tissue_hole_area,
        )

        self.tissue_cleaned_mask = np.asarray(cleaned, dtype=bool)

        # Keep largest connected tissue component.
        tissue_labels = measure.label(cleaned)
        tissue_regions = measure.regionprops(tissue_labels)

        if not tissue_regions:
            raise RuntimeError(
                "No tissue region remained after tissue-mask processing."
            )

        largest = max(
            tissue_regions,
            key=lambda region: region.area,
        )

        tissue_mask = (
            tissue_labels == largest.label
        )

        # --------------------------------------------------------------
        # STEP 4 — erode tissue
        # --------------------------------------------------------------
        tissue_mask_eroded = tissue_mask.copy()

        for _ in range(int(tissue_erosion_iterations)):
            tissue_mask_eroded = binary_erosion(
                tissue_mask_eroded,
                footprint=morph_disk(
                    int(tissue_erosion_radius)
                ),
            )

        # --------------------------------------------------------------
        # STEP 5 — fluorescence threshold inside eroded tissue
        # --------------------------------------------------------------
        object_candidate_mask = (
            img > object_threshold
        )

        object_candidate_mask[
            tissue_mask_eroded == 0
        ] = 0

        self.object_candidate_mask = (
            np.asarray(object_candidate_mask, dtype=bool)
        )

        candidate_labels = label(
            self.object_candidate_mask
        )

        candidate_props = regionprops(
            candidate_labels
        )

        # --------------------------------------------------------------
        # STEP 6 — area / roundness filtering
        # Preserve original OR logic exactly.
        # --------------------------------------------------------------
        filtered_mask = np.zeros_like(
            object_candidate_mask,
            dtype=bool,
        )

        for region in candidate_props:
            area = region.area

            if region.perimeter > 0:
                circularity = (
                    4.0
                    * np.pi
                    * area
                    / (region.perimeter ** 2)
                )
            else:
                circularity = 0.0

            if (
                object_min_size
                <= area
                <= object_max_size
                or circularity >= min_roundness
            ):
                filtered_mask[
                    candidate_labels == region.label
                ] = True

        # --------------------------------------------------------------
        # STEP 7 — convert physical distances to pixels
        # --------------------------------------------------------------
        pixel_size_um = float(pixel_size_um)
        circle_radius_um = float(circle_radius_um)

        circle_radius_px = int(
            round(
                circle_radius_um
                / pixel_size_um
            )
        )

        circle_radius_px = max(
            1,
            circle_radius_px,
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
        # STEP 8 — connected components + centroids
        # --------------------------------------------------------------
        filtered_labels = label(
            filtered_mask > 0
        )

        filtered_props = regionprops(
            filtered_labels
        )

        if filtered_props:
            centroids_before_nn = np.asarray(
                [
                    prop.centroid
                    for prop in filtered_props
                ],
                dtype=float,
            ).reshape(-1, 2)
        else:
            centroids_before_nn = np.empty(
                (0, 2),
                dtype=float,
            )

        self.centroids_before_nn = (
            centroids_before_nn.copy()
        )

        # --------------------------------------------------------------
        # STEP 9 — optional nearest-neighbour filtering
        # --------------------------------------------------------------
        centroids_after_nn = (
            centroids_before_nn.copy()
        )

        if (
            use_nearest_neighbor_filter
            and len(centroids_after_nn) >= 2
        ):
            nbrs = NearestNeighbors(
                n_neighbors=2
            )

            nbrs.fit(
                centroids_after_nn
            )

            distances, _ = nbrs.kneighbors(
                centroids_after_nn
            )

            # Remove self-distance; one non-self nearest neighbour remains.
            nearest_neighbor_distance = (
                distances[:, 1]
            )

            keep = (
                (
                    nearest_neighbor_distance
                    >= min_nn_px
                )
                & (
                    nearest_neighbor_distance
                    <= max_nn_px
                )
            )

            centroids_after_nn = (
                centroids_after_nn[
                    keep
                ]
            )

        # If only one candidate exists, there is no non-self NN distance.
        # Keep it rather than crashing. This affects only this edge case.
        self.centroids_after_nn = (
            centroids_after_nn.copy()
        )

        # --------------------------------------------------------------
        # STEP 10 — random cap / selection
        # --------------------------------------------------------------
        selected_centroids = (
            centroids_after_nn.copy()
        )

        rng = np.random.default_rng(
            random_seed
        )

        if (
            max_objects is not None
            and len(selected_centroids)
            > int(max_objects)
        ):
            idx = rng.choice(
                len(selected_centroids),
                int(max_objects),
                replace=False,
            )

            selected_centroids = (
                selected_centroids[idx]
            )

        self.selected_centroids_rc = (
            selected_centroids.copy()
        )

        # --------------------------------------------------------------
        # STEP 11 — create final fixed-radius circular sampling mask
        # --------------------------------------------------------------
        final_mask = np.zeros(
            filtered_labels.shape,
            dtype=bool,
        )

        for row, col in selected_centroids:
            rr, cc = disk(
                (row, col),
                circle_radius_px,
                shape=final_mask.shape,
            )

            final_mask[
                rr,
                cc,
            ] = True

        # --------------------------------------------------------------
        # STEP 12 — reproduce original binary mask conversion
        # --------------------------------------------------------------
        final_mask_for_output = (
            final_mask.astype(
                np.float32
            )
        )

        final_mask_for_output[
            final_mask_for_output == 0
        ] = np.nan

        if np.any(
            np.isfinite(
                final_mask_for_output
            )
        ):
            final_mask_for_output = (
                final_mask_for_output
                / np.nanmax(
                    final_mask_for_output
                )
            )

        binary_mask = (
            np.nan_to_num(
                final_mask_for_output
            )
            > 0
        )

        binary_mask = (
            binary_mask.astype(
                np.uint8
            )
            * 255
        )

        # Reproduce original RGB -> grayscale output construction.
        rgb = np.stack(
            [binary_mask] * 3,
            axis=-1,
        )

        out_arr = rgb[
            np.newaxis,
            ...,
        ]

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
        # STEP 13 — output geometry
        # --------------------------------------------------------------
        if preserve_source_geometry:
            if self.source.GetDimension() != 3:
                raise ValueError(
                    "preserve_source_geometry=True requires a 3D "
                    "single-slice source image."
                )

            if (
                tuple(out_img.GetSize())
                != tuple(self.source.GetSize())
            ):
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

        # Keep original no-op shrink factors.
        shrink = sitk.ShrinkImageFilter()
        shrink.SetShrinkFactors(
            [1, 1, 1]
        )
        out_img = shrink.Execute(
            out_img
        )

        # --------------------------------------------------------------
        # Store state
        # --------------------------------------------------------------
        self.tissue_mask = tissue_mask
        self.tissue_mask_eroded = (
            tissue_mask_eroded
        )
        self.filtered_object_mask = (
            filtered_mask
        )
        self.centroids_rc = (
            selected_centroids.copy()
        )
        self.final_mask = final_mask
        self.binary_mask = binary_mask
        self.mask_image = out_img

        self.stage_counts = {
            "tissue_components_after_cleanup": int(
                len(tissue_regions)
            ),
            "candidate_objects_after_threshold": int(
                len(candidate_props)
            ),
            "objects_after_area_roundness_filter": int(
                len(filtered_props)
            ),
            "objects_before_nn_filter": int(
                len(centroids_before_nn)
            ),
            "objects_after_nn_filter": int(
                len(centroids_after_nn)
            ),
            "objects_after_random_cap": int(
                len(selected_centroids)
            ),
            "final_sampling_rois": int(
                len(selected_centroids)
            ),
            "final_positive_pixels": int(
                np.count_nonzero(final_mask)
            ),
        }

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
            "use_nearest_neighbor_filter": bool(
                use_nearest_neighbor_filter
            ),
            "max_objects": (
                None
                if max_objects is None
                else int(max_objects)
            ),
            "circle_radius_um": circle_radius_um,
            "circle_radius_px": circle_radius_px,
            "random_seed": random_seed,
            "preserve_source_geometry": bool(
                preserve_source_geometry
            ),
            "selected_objects": int(
                len(selected_centroids)
            ),
        }

        self._build_summary()

        if self.save_figures or self.show_figures:
            self._plot_qc_figures()

        # Rebuild after plotting so saved figure paths are included.
        self._build_summary()

        if self.save_summary:
            self._write_summary_json()

        return self

    # =================================================================
    # QC figure helpers
    # =================================================================

    def _finish_figure(
        self,
        fig,
        name,
    ):
        self._figure_counter += 1

        if self.save_figures:
            self.figures_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            safe_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(name),
            ).strip("_")

            path = (
                self.figures_dir
                / (
                    f"{self._figure_counter:03d}_"
                    f"{safe_name}.png"
                )
            )

            fig.savefig(
                path,
                dpi=200,
                bbox_inches="tight",
            )

            self.saved_figures.append(
                str(path)
            )

        if self.show_figures:
            plt.show()

        plt.close(fig)

    def _scatter_centroids(
        self,
        ax,
        centroids,
        title,
    ):
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="gray",
        )

        if len(centroids):
            ax.scatter(
                centroids[:, 1],
                centroids[:, 0],
                s=18,
                facecolors="none",
                edgecolors="cyan",
                linewidths=0.8,
            )

        ax.set_title(
            f"{title}\nN={len(centroids)}"
        )
        ax.axis("off")

    def _plot_qc_figures(self):
        if self.mask_image is None:
            raise RuntimeError(
                "Generate() must finish before plotting QC figures."
            )

        # 001 — raw IF
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="magma",
        )
        ax.set_title(
            f"{self.sample_name} — raw IF"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "raw_IF",
        )

        # 002 — smoothed IF
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.blur
            ),
            cmap="magma",
        )
        ax.set_title(
            f"{self.sample_name} — Gaussian-smoothed IF"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "smoothed_IF",
        )

        # 003 — initial tissue threshold
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            self.tissue_threshold_mask,
            cmap="gray",
        )
        ax.set_title(
            f"{self.sample_name} — initial tissue threshold"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "initial_tissue_threshold",
        )

        # 004 — final largest tissue component
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="gray",
        )
        if np.any(self.tissue_mask):
            ax.contour(
                self.tissue_mask.astype(float),
                levels=[0.5],
                colors="lime",
                linewidths=0.9,
            )
        ax.set_title(
            f"{self.sample_name} — retained tissue mask"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "retained_tissue_mask",
        )

        # 005 — eroded tissue
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="gray",
        )
        if np.any(self.tissue_mask_eroded):
            ax.contour(
                self.tissue_mask_eroded.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=0.9,
            )
        ax.set_title(
            f"{self.sample_name} — eroded tissue mask"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "eroded_tissue_mask",
        )

        # 006 — all threshold candidates
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="gray",
        )
        if np.any(self.object_candidate_mask):
            ax.contour(
                self.object_candidate_mask.astype(float),
                levels=[0.5],
                colors="yellow",
                linewidths=0.7,
            )
        ax.set_title(
            f"{self.sample_name} — fluorescence candidates after threshold\n"
            f"N={self.stage_counts['candidate_objects_after_threshold']}"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "threshold_candidates",
        )

        # 007 — after area/roundness filter
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="gray",
        )
        if np.any(self.filtered_object_mask):
            ax.contour(
                self.filtered_object_mask.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=0.8,
            )
        ax.set_title(
            f"{self.sample_name} — candidates after area/roundness filter\n"
            f"N={self.stage_counts['objects_after_area_roundness_filter']}"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "filtered_candidates",
        )

        # 008 — centroids before NN
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        self._scatter_centroids(
            ax,
            self.centroids_before_nn,
            f"{self.sample_name} — centroids before NN filter",
        )
        self._finish_figure(
            fig,
            "centroids_before_nn_filter",
        )

        # 009 — centroids after NN
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        self._scatter_centroids(
            ax,
            self.centroids_after_nn,
            f"{self.sample_name} — centroids after NN filter",
        )
        self._finish_figure(
            fig,
            "centroids_after_nn_filter",
        )

        # 010 — selected centroids after cap
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        self._scatter_centroids(
            ax,
            self.selected_centroids_rc,
            f"{self.sample_name} — final selected centroids",
        )
        self._finish_figure(
            fig,
            "selected_centroids",
        )

        # 011 — final circular mask
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            self.final_mask,
            cmap="gray",
        )
        ax.set_title(
            f"{self.sample_name} — final {self.parameters['circle_radius_um']:.1f} µm "
            f"sampling mask\nN={self.stage_counts['final_sampling_rois']}"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "final_sampling_mask",
        )

        # 012 — final overlay
        fig, ax = plt.subplots(
            figsize=(9, 11)
        )
        ax.imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="magma",
        )
        if np.any(self.final_mask):
            ax.contour(
                self.final_mask.astype(float),
                levels=[0.5],
                colors="lime",
                linewidths=0.9,
            )
        ax.set_title(
            f"{self.sample_name} — IF + final sampling-mask contour"
        )
        ax.axis("off")
        self._finish_figure(
            fig,
            "IF_plus_final_mask",
        )

        # 013 — compact 2x3 overview
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(16, 10),
            constrained_layout=True,
        )

        axes[0, 0].imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="magma",
        )
        axes[0, 0].set_title("Raw IF")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(
            self.tissue_mask,
            cmap="gray",
        )
        axes[0, 1].set_title("Retained tissue")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(
            self.tissue_mask_eroded,
            cmap="gray",
        )
        axes[0, 2].set_title("Eroded tissue")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(
            self.object_candidate_mask,
            cmap="gray",
        )
        axes[1, 0].set_title(
            "Threshold candidates\n"
            f"N={self.stage_counts['candidate_objects_after_threshold']}"
        )
        axes[1, 0].axis("off")

        axes[1, 1].imshow(
            self.filtered_object_mask,
            cmap="gray",
        )
        axes[1, 1].set_title(
            "After morphology filter\n"
            f"N={self.stage_counts['objects_after_area_roundness_filter']}"
        )
        axes[1, 1].axis("off")

        axes[1, 2].imshow(
            _display_normalize(
                self.image_array
            ),
            cmap="magma",
        )
        if np.any(self.final_mask):
            axes[1, 2].contour(
                self.final_mask.astype(float),
                levels=[0.5],
                colors="lime",
                linewidths=0.8,
            )
        axes[1, 2].set_title(
            "Final mask\n"
            f"N={self.stage_counts['final_sampling_rois']}"
        )
        axes[1, 2].axis("off")

        fig.suptitle(
            f"{self.sample_name} — plaque-mask generation QC",
            fontsize=14,
        )

        self._finish_figure(
            fig,
            "mask_generation_summary",
        )

    # =================================================================
    # Summary / public getters
    # =================================================================

    def _build_summary(self):
        self.summary = {
            "sample": self.sample_name,
            "stage_counts": dict(
                self.stage_counts
            ),
            "parameters": dict(
                self.parameters
            ),
            "outputs": {
                "mask_nrrd": None,
                "figures_dir": (
                    str(self.figures_dir)
                    if (
                        self.save_figures
                        and self.figures_dir is not None
                    )
                    else None
                ),
                "summary_json": (
                    str(self.summary_json_path)
                    if (
                        self.save_summary
                        and self.summary_json_path is not None
                    )
                    else None
                ),
                "saved_figures": list(
                    self.saved_figures
                ),
            },
        }

    def _write_summary_json(self):
        if self.summary_json_path is None:
            raise RuntimeError(
                "No output_dir was configured."
            )

        self._build_summary()

        self.summary_json_path.write_text(
            json.dumps(
                self.summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    def GetMaskImage(self):
        if self.mask_image is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return sitk.Image(
            self.mask_image
        )

    def GetArray(
        self,
        squeeze=True,
    ):
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

    def GetCandidateMask(self):
        if self.object_candidate_mask is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.object_candidate_mask.copy()

    def GetFilteredObjectMask(self):
        if self.filtered_object_mask is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        return self.filtered_object_mask.copy()

    def GetCentroids(self):
        return self.centroids_rc.copy()

    def GetCentroidsBeforeNN(self):
        return self.centroids_before_nn.copy()

    def GetCentroidsAfterNN(self):
        return self.centroids_after_nn.copy()

    def GetStageCounts(self):
        return dict(
            self.stage_counts
        )

    def GetParameters(self):
        return dict(
            self.parameters
        )

    def GetSummary(self):
        if self.summary is None:
            raise RuntimeError(
                "Generate() must be called first."
            )
        self._build_summary()
        return dict(
            self.summary
        )

    def GetSavedFigures(self):
        return [
            Path(path)
            for path in self.saved_figures
        ]

    def WriteNRRD(
        self,
        output_path,
        use_compression=False,
    ):
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sitk.WriteImage(
            self.GetMaskImage(),
            str(output_path),
            bool(use_compression),
        )

        if self.summary is not None:
            self._build_summary()
            self.summary["outputs"][
                "mask_nrrd"
            ] = str(output_path)

            if self.save_summary:
                self.summary_json_path.write_text(
                    json.dumps(
                        self.summary,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        return output_path
