from pathlib import Path
import json
import re
from typing import Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import morphology


PathLike = Union[str, Path]
ImageLike = Union[PathLike, sitk.Image]


def _as_image(source: ImageLike) -> sitk.Image:
    """Load a SimpleITK image from a path or copy an existing image."""
    if isinstance(source, sitk.Image):
        return sitk.Image(source)

    if isinstance(source, (str, Path)):
        return sitk.ReadImage(str(source))

    raise TypeError(
        "Expected a SimpleITK.Image or image filename."
    )


def _squeeze2d(image: sitk.Image) -> np.ndarray:
    """Return a 2D NumPy array from a scalar 2D/single-slice 3D image."""
    arr = np.squeeze(
        sitk.GetArrayFromImage(image)
    )

    if arr.ndim != 2:
        raise ValueError(
            "MIRIFAlignmentQC requires a scalar 2D image or "
            f"single-slice 3D image; got shape {arr.shape}."
        )

    return arr


def _same_grid(
    a: sitk.Image,
    b: sitk.Image,
    atol: float = 1e-9,
) -> bool:
    """Check Size, Spacing, Origin and Direction."""
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(
            a.GetSpacing(),
            b.GetSpacing(),
            atol=atol,
        )
        and np.allclose(
            a.GetOrigin(),
            b.GetOrigin(),
            atol=atol,
        )
        and np.allclose(
            a.GetDirection(),
            b.GetDirection(),
            atol=atol,
        )
    )


def _resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    output_pixel_type: int,
    default_value: float = 0,
) -> sitk.Image:
    """
    Sample an already physically positioned image onto the reference grid.

    This does NOT estimate a new registration transform.
    """
    identity = sitk.Transform(
        reference.GetDimension(),
        sitk.sitkIdentity,
    )

    return sitk.Resample(
        moving,
        reference,
        identity,
        interpolator,
        default_value,
        output_pixel_type,
    )


def _robust_normalize(
    arr: np.ndarray,
    valid: Optional[np.ndarray] = None,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Percentile normalization for display/structural comparison."""
    arr = np.asarray(
        arr,
        dtype=np.float64,
    )

    if valid is None:
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = (
            np.asarray(valid, dtype=bool)
            & np.isfinite(arr)
        )

    out = np.zeros_like(
        arr,
        dtype=np.float64,
    )

    if not np.any(valid_mask):
        return out

    lo, hi = np.percentile(
        arr[valid_mask],
        [low, high],
    )

    if hi <= lo:
        return out

    out = (arr - lo) / (hi - lo)
    out = np.clip(
        out,
        0.0,
        1.0,
    )
    out[~valid_mask] = 0.0

    return out


def _checkerboard(
    a: np.ndarray,
    b: np.ndarray,
    block: int = 40,
) -> np.ndarray:
    """Create a checkerboard image for visual registration QC."""
    if a.shape != b.shape:
        raise ValueError(
            f"Checkerboard requires same shape: "
            f"{a.shape} vs {b.shape}"
        )

    yy, xx = np.indices(a.shape)

    choose_a = (
        ((yy // block) + (xx // block)) % 2
        == 0
    )

    return np.where(
        choose_a,
        a,
        b,
    )


def _gradient_structure(
    arr: np.ndarray,
    valid: np.ndarray,
    gaussian_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a normalized image and gradient-magnitude image.

    Gradient structure is used because BF, MIR and IF do not share
    the same raw intensity contrast.
    """
    norm = _robust_normalize(
        arr,
        valid=valid,
    )

    smooth = ndi.gaussian_filter(
        norm,
        sigma=gaussian_sigma,
        mode="nearest",
    )

    gy = ndi.sobel(
        smooth,
        axis=0,
        mode="nearest",
    )

    gx = ndi.sobel(
        smooth,
        axis=1,
        mode="nearest",
    )

    grad = np.hypot(
        gx,
        gy,
    )

    vals = grad[valid]

    if vals.size:
        hi = np.percentile(
            vals,
            99,
        )

        if hi > 0:
            grad = np.clip(
                grad / hi,
                0.0,
                1.0,
            )

    grad[~valid] = 0.0

    return norm, grad


def _make_edge_map(
    grad: np.ndarray,
    valid: np.ndarray,
    percentile: float = 80.0,
    dilation_radius: int = 1,
) -> np.ndarray:
    """Create a binary strong-edge map from gradient magnitude."""
    vals = grad[
        valid
        & np.isfinite(grad)
    ]

    if vals.size == 0:
        return np.zeros_like(
            valid,
            dtype=bool,
        )

    threshold = np.percentile(
        vals,
        percentile,
    )

    edge = (
        (grad >= threshold)
        & valid
    )

    if dilation_radius > 0:
        edge = morphology.dilation(
            edge,
            footprint=morphology.disk(
                dilation_radius
            ),
        )

    return edge


def _crop_to_support(
    fixed: np.ndarray,
    moving: np.ndarray,
    moving_support: np.ndarray,
    margin_px: int = 10,
):
    """Crop both images around the non-zero moving-modality support."""
    rows, cols = np.nonzero(
        moving_support
    )

    if len(rows) == 0:
        raise RuntimeError(
            "Moving modality has no non-zero support."
        )

    y0 = max(
        0,
        int(rows.min()) - margin_px,
    )
    y1 = min(
        fixed.shape[0],
        int(rows.max()) + margin_px + 1,
    )
    x0 = max(
        0,
        int(cols.min()) - margin_px,
    )
    x1 = min(
        fixed.shape[1],
        int(cols.max()) + margin_px + 1,
    )

    return (
        fixed[y0:y1, x0:x1],
        moving[y0:y1, x0:x1],
        moving_support[y0:y1, x0:x1],
        (x0, x1, y0, y1),
    )


def _downsample_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    support: np.ndarray,
    spacing_xy_um: Tuple[float, float],
    target_um: float,
):
    """Downsample a pair to an approximate target physical sampling."""
    sx_um, sy_um = spacing_xy_um

    zoom_y = min(
        1.0,
        sy_um / target_um,
    )
    zoom_x = min(
        1.0,
        sx_um / target_um,
    )

    fixed_ds = ndi.zoom(
        fixed,
        (zoom_y, zoom_x),
        order=1,
    )

    moving_ds = ndi.zoom(
        moving,
        (zoom_y, zoom_x),
        order=1,
    )

    support_ds = ndi.zoom(
        support.astype(np.uint8),
        (zoom_y, zoom_x),
        order=0,
    ).astype(bool)

    eff_sy_um = (
        sy_um / zoom_y
    )
    eff_sx_um = (
        sx_um / zoom_x
    )

    return (
        fixed_ds,
        moving_ds,
        support_ds,
        (eff_sx_um, eff_sy_um),
    )


def _erode_support_physical(
    support: np.ndarray,
    spacing_xy_um: Tuple[float, float],
    erosion_um: float,
) -> np.ndarray:
    """
    Erode support to reduce field-of-view boundary influence
    in structural registration scoring.
    """
    sx_um, sy_um = spacing_xy_um

    mean_um = 0.5 * (
        sx_um + sy_um
    )

    radius_px = max(
        1,
        int(
            round(
                erosion_um / mean_um
            )
        ),
    )

    eroded = morphology.erosion(
        support,
        footprint=morphology.disk(
            radius_px
        ),
    )

    # Do not over-erode tiny supports.
    if eroded.sum() < (
        0.25 * support.sum()
    ):
        return support.copy()

    return eroded


def _shifted_overlap_slices(
    shape: Tuple[int, int],
    dy: int,
    dx: int,
):
    """Return overlapping fixed/moving slices for an integer shift."""
    h, w = shape

    if dy >= 0:
        fy = slice(
            dy,
            h,
        )
        my = slice(
            0,
            h - dy,
        )
    else:
        fy = slice(
            0,
            h + dy,
        )
        my = slice(
            -dy,
            h,
        )

    if dx >= 0:
        fx = slice(
            dx,
            w,
        )
        mx = slice(
            0,
            w - dx,
        )
    else:
        fx = slice(
            0,
            w + dx,
        )
        mx = slice(
            -dx,
            w,
        )

    return (
        (fy, fx),
        (my, mx),
    )


def _pearson_ncc(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Pearson normalized cross-correlation.

    Range:
        -1 = opposite structure
         0 = no linear correlation
        +1 = perfect structural correlation
    """
    a = np.asarray(
        a,
        dtype=np.float64,
    )
    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if a.size < 100:
        return np.nan

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(a, b)
        / denom
    )


def _edge_dice(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Dice overlap between binary strong-edge maps.

    Range:
        0 = no overlap
        1 = perfect overlap
    """
    a = np.asarray(
        a,
        dtype=bool,
    )
    b = np.asarray(
        b,
        dtype=bool,
    )

    total = (
        int(a.sum())
        + int(b.sum())
    )

    if total == 0:
        return np.nan

    intersection = int(
        np.logical_and(
            a,
            b,
        ).sum()
    )

    return float(
        2.0 * intersection
        / total
    )


def _score_transform(
    fixed_grad: np.ndarray,
    moving_grad_rot: np.ndarray,
    fixed_edge: np.ndarray,
    moving_edge_rot: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask_rot: np.ndarray,
    dy: int,
    dx: int,
) -> Dict[str, float]:
    """
    Score one residual transform.

    Final score:
        0.70 * normalized Gradient NCC
        +
        0.30 * Edge Dice
    """
    fixed_slice, moving_slice = (
        _shifted_overlap_slices(
            fixed_grad.shape,
            dy,
            dx,
        )
    )

    overlap = (
        fixed_mask[fixed_slice]
        & moving_mask_rot[moving_slice]
    )

    if overlap.sum() < 1000:
        return {
            "score": np.nan,
            "ncc": np.nan,
            "edge_dice": np.nan,
            "overlap_pixels": int(
                overlap.sum()
            ),
        }

    fixed_gradient_values = (
        fixed_grad[fixed_slice][overlap]
    )

    moving_gradient_values = (
        moving_grad_rot[moving_slice][overlap]
    )

    ncc = _pearson_ncc(
        fixed_gradient_values,
        moving_gradient_values,
    )

    fixed_edges = (
        fixed_edge[fixed_slice][overlap]
    )

    moving_edges = (
        moving_edge_rot[moving_slice][overlap]
    )

    edge_dice = _edge_dice(
        fixed_edges,
        moving_edges,
    )

    if not np.isfinite(ncc):
        ncc = -1.0

    if not np.isfinite(edge_dice):
        edge_dice = 0.0

    # Convert NCC from [-1, 1] to [0, 1].
    ncc_01 = 0.5 * (
        ncc + 1.0
    )

    score = (
        0.70 * ncc_01
        + 0.30 * edge_dice
    )

    return {
        "score": float(score),
        "ncc": float(ncc),
        "edge_dice": float(edge_dice),
        "overlap_pixels": int(
            overlap.sum()
        ),
    }


def _shift_image_geometry(
    image: sitk.Image,
    dx_um: float = 0.0,
    dy_um: float = 0.0,
    dz_um: float = 0.0,
    axis_reference: Optional[
        sitk.Image
    ] = None,
) -> sitk.Image:
    """
    Translate an image by changing Origin only.

    Pixel values are preserved exactly.

    dx/dy/dz are expressed along the reference image index axes.
    The reference Direction matrix converts that shift into physical space.
    """
    if axis_reference is None:
        axis_reference = image

    if (
        image.GetDimension()
        != axis_reference.GetDimension()
    ):
        raise ValueError(
            "image and axis_reference must have the same dimension."
        )

    shifted = sitk.Image(
        image
    )

    dim = image.GetDimension()

    direction = np.asarray(
        axis_reference.GetDirection(),
        dtype=float,
    ).reshape(
        dim,
        dim,
    )

    shift_index_axes_mm = np.zeros(
        dim,
        dtype=float,
    )

    shift_index_axes_mm[0] = (
        float(dx_um) / 1000.0
    )

    if dim >= 2:
        shift_index_axes_mm[1] = (
            float(dy_um) / 1000.0
        )

    if dim >= 3:
        shift_index_axes_mm[2] = (
            float(dz_um) / 1000.0
        )

    physical_shift_mm = (
        direction
        @ shift_index_axes_mm
    )

    old_origin = np.asarray(
        image.GetOrigin(),
        dtype=float,
    )

    new_origin = (
        old_origin
        + physical_shift_mm
    )

    shifted.SetOrigin(
        tuple(new_origin)
    )

    return shifted


def _rigid_adjust_image_geometry(
    image: sitk.Image,
    reference: sitk.Image,
    dx_um: float = 0.0,
    dy_um: float = 0.0,
    angle_deg: float = 0.0,
    center_physical: Optional[
        Tuple[float, ...]
    ] = None,
) -> sitk.Image:
    """
    Apply a small in-plane rigid correction by changing image geometry only.

    The pixel array is NOT resampled here.

    Parameters
    ----------
    image
        Image whose physical geometry will be adjusted.

    reference
        Fixed reference defining the x/y index axes used by the QC search.
        For direct MIR↔IF refinement this is the MIR image.

    dx_um, dy_um
        Translation in micrometers along the reference x/y index axes.

    angle_deg
        In-plane rotation in degrees using the SAME sign convention as the
        NumPy/SciPy QC search:
            positive = counter-clockwise in the displayed image.

    center_physical
        Physical-space rotation center.  If None, the physical center of the
        reference image is used.

    Returns
    -------
    SimpleITK.Image
        Copy of ``image`` with updated Origin and Direction.

    Notes
    -----
    The transform applied to each original physical point p is

        p_new = c + R (p - c) + t

    where c is the physical rotation center, R is the in-plane rotation
    expressed in physical coordinates, and t is the translation.

    Size, Spacing, and the original pixel values are preserved exactly.
    """
    if (
        image.GetDimension()
        != reference.GetDimension()
    ):
        raise ValueError(
            "image and reference must have the same dimension."
        )

    dim = image.GetDimension()

    if dim not in (2, 3):
        raise ValueError(
            "Rigid in-plane refinement currently supports 2D images "
            "or single-slice 3D images."
        )

    if dim == 3:
        if (
            image.GetSize()[2] != 1
            or reference.GetSize()[2] != 1
        ):
            raise ValueError(
                "3D rigid refinement is restricted to single-slice images."
            )

    corrected = sitk.Image(
        image
    )

    # Reference Direction maps reference-index-axis vectors into
    # physical-space vectors.
    d_ref = np.asarray(
        reference.GetDirection(),
        dtype=float,
    ).reshape(
        dim,
        dim,
    )

    d_img = np.asarray(
        image.GetDirection(),
        dtype=float,
    ).reshape(
        dim,
        dim,
    )

    theta = np.deg2rad(
        float(angle_deg)
    )

    cth = float(
        np.cos(theta)
    )
    sth = float(
        np.sin(theta)
    )

    # scipy.ndimage.rotate positive angle = counter-clockwise visually.
    # In array/index coordinates y increases downward, therefore the
    # corresponding x/y coordinate transform is:
    #
    #   x' =  cos(theta) * x + sin(theta) * y
    #   y' = -sin(theta) * x + cos(theta) * y
    #
    # This matrix exactly represents that sign convention.
    r_index = np.eye(
        dim,
        dtype=float,
    )

    r_index[0, 0] = cth
    r_index[0, 1] = sth
    r_index[1, 0] = -sth
    r_index[1, 1] = cth

    # Convert the reference-index-axis rotation into physical coordinates.
    #
    # Direction matrices are orthonormal, so inverse(D) == transpose(D).
    r_physical = (
        d_ref
        @ r_index
        @ d_ref.T
    )

    shift_index_mm = np.zeros(
        dim,
        dtype=float,
    )

    shift_index_mm[0] = (
        float(dx_um) / 1000.0
    )

    shift_index_mm[1] = (
        float(dy_um) / 1000.0
    )

    # Translation is specified along MIR/reference index axes.
    shift_physical_mm = (
        d_ref
        @ shift_index_mm
    )

    if center_physical is None:
        center_index = [
            0.5 * (
                float(size) - 1.0
            )
            for size in reference.GetSize()
        ]

        center_physical_arr = np.asarray(
            reference.TransformContinuousIndexToPhysicalPoint(
                tuple(center_index)
            ),
            dtype=float,
        )
    else:
        center_physical_arr = np.asarray(
            center_physical,
            dtype=float,
        )

        if center_physical_arr.shape != (
            dim,
        ):
            raise ValueError(
                "center_physical must have one coordinate per image dimension."
            )

    old_origin = np.asarray(
        image.GetOrigin(),
        dtype=float,
    )

    new_origin = (
        center_physical_arr
        + r_physical
        @ (
            old_origin
            - center_physical_arr
        )
        + shift_physical_mm
    )

    new_direction = (
        r_physical
        @ d_img
    )

    corrected.SetOrigin(
        tuple(
            float(v)
            for v in new_origin
        )
    )

    corrected.SetDirection(
        tuple(
            float(v)
            for v in new_direction.ravel()
        )
    )

    return corrected


class MIRIFAlignmentQC:
    """
    Quality-control and small residual-refinement workflow for pre-aligned MIR,
    fluorescence, plaque-mask, and Brightfield images.

    This class does NOT perform the initial global registration. It expects pre-aligned inputs.

    It is intended for the workflow:

        initial alignment
            ↓
        Registration QC using Brightfield
            ↓
        Direct MIR ↔ IF residual estimation
            ↓
        compare translation-only vs rigid refinement
            ↓
        apply rotation only if it adds sufficient score beyond translation
            ↓
        closed-loop validation
            ↓
        transfer accepted IF + plaque mask onto the MIR pixel grid

    Scientific roles
    ----------------
    brightfield
        Independent anatomical reference used to judge whether the supplied
        pre-alignment is globally plausible.

    mir_aligned
        MIR registration image (for example 1650 cm⁻¹). MIR is kept FIXED
        during the residual-refinement step.

    if_aligned
        Pre-aligned fluorescence image. IF may receive a small,
        small rigid residual correction if all safety gates pass.

    mask_aligned
        Plaque mask derived from IF. It must have the same geometry as IF and
        always receives exactly the same accepted correction as IF.

    Structural similarity score
    ---------------------------
    Raw BF, MIR and IF brightness cannot be compared directly because the
    modalities have different contrast. Therefore the QC uses structural
    information:

        score =
            0.70 * normalized Gradient NCC
            +
            0.30 * Edge Dice

    Gradient NCC
        Pearson normalized cross-correlation of gradient-magnitude images.
        Range approximately -1 to +1. Higher means similar image structure.

    Edge Dice
        Dice overlap of strong binary edge maps.
        Range 0 to 1. Higher means stronger edge overlap.

    Fail-fast philosophy
    --------------------
    The local optimizer is only allowed to refine an already reasonable initial alignment. If the initial error is too large, structural agreement is too
    weak, the optimum hits a search boundary, or the post-correction result is
    unstable, the class stops rather than creating a training label.

    Public parameters
    -----------------
    plaque_radius_um : float, default 40
        Biological plaque ROI radius. It is also used to scale the maximum
        acceptable direct residual shift.

    plaque_zoom_count : int, default 6
        Number of plaque regions displayed in the final zoom-validation figure.
        Set to 0 to disable plaque zoom panels.

    plaque_zoom_selection : {"largest", "random", "mixed"}, default "largest"
        Strategy used to choose plaque regions for the final zoom-validation
        figure. ``largest`` preserves the historical behavior, ``random``
        draws a seeded random subset, and ``mixed`` shows approximately half
        largest plaques and half seeded-random plaques from the remainder.

    plaque_zoom_seed : int or None, default 42
        Random seed used only for plaque-zoom QC selection. The registration
        and refinement algorithms are deterministic and do not use this seed.
        Set to None for a different random subset each run.

    auto_refine : bool, default True
        If True, apply a small safe rigid correction (translation + rotation) to IF + mask when all gates pass.
        MIR and BF are never moved.

    save_figures : bool, default True
        Save registration/QC figures in output_dir / "figures".

    show_figures : bool, default False
        Also display figures interactively.

    gross_max_shift_um : float or None, default None
        Maximum BF-based residual shift allowed before the class says:
        "return to initial alignment". None means 2 * plaque_radius_um.

    gross_max_rotation_deg : float, default 1.0
        Maximum BF-based residual rotation allowed by the gross alignment gate.

    min_gradient_ncc : float, default 0.10
        Minimum Gradient NCC used by both the gross and direct structural gates.

    min_edge_dice : float, default 0.40
        Minimum Edge Dice used by both the gross and direct structural gates.

    direct_max_shift_um : float or None, default None
        Maximum direct MIR↔IF residual translation. None means plaque_radius_um.

    direct_max_rotation_deg : float, default 0.5
        Maximum direct residual rotation eligible for automatic rigid refinement.

    min_score_gain : float, default 0.003
        Minimum structural-score improvement required before automatically
        applying any correction.

    rotation_min_extra_gain : float, default 0.003
        Minimum additional score improvement that the rigid solution must
        provide beyond the best translation-only solution before rotation is
        applied.

    post_max_residual_shift_um : float, default 10
        Maximum residual shift allowed after correction.

    post_max_rotation_deg : float, default 0.25
        Maximum residual rotation allowed after correction.

    post_max_score_gain : float, default 0.003
        Maximum remaining possible improvement after correction.

    Outputs
    -------
    Accepted image outputs:
        <sample>_IF_refined_to_MIR.nrrd
        <sample>_IF_mask_refined_to_MIR.nrrd
        <sample>_IF_refined_on_MIR_grid.nrrd
        <sample>_IF_mask_refined_on_MIR_grid.nrrd

    QC summary:
        <sample>_MIR_IF_alignment_QC_summary.json

    Figures:
        output_dir / "figures" / *.png

    No CSV, TXT report, or manifest is created. The JSON file is the single
    summary/report artifact.

    Notes
    -----
    The defaults intentionally follow the latest validated notebook workflow.
    Search-resolution parameters remain internal because they are algorithm
    details, not routine user controls.
    """

    def __init__(
        self,
        brightfield: ImageLike,
        mir_aligned: ImageLike,
        if_aligned: ImageLike,
        mask_aligned: ImageLike,
        output_dir: PathLike,
        sample_name: str = "sample",
        plaque_radius_um: float = 40.0,
        plaque_zoom_count: int = 6,
        plaque_zoom_selection: str = "largest",
        plaque_zoom_seed: Optional[int] = 42,
        auto_refine: bool = True,
        save_figures: bool = True,
        show_figures: bool = False,
        gross_max_shift_um: Optional[float] = None,
        gross_max_rotation_deg: float = 1.0,
        min_gradient_ncc: float = 0.10,
        min_edge_dice: float = 0.40,
        direct_max_shift_um: Optional[float] = None,
        direct_max_rotation_deg: float = 0.5,
        min_score_gain: float = 0.003,
        rotation_min_extra_gain: float = 0.003,
        post_max_residual_shift_um: float = 10.0,
        post_max_rotation_deg: float = 0.25,
        post_max_score_gain: float = 0.003,
    ):
        """
        Create a QC/refinement object from four explicit pre-aligned inputs.

        Parameters
        ----------
        brightfield
            Fixed Brightfield reference image.

        mir_aligned
            Pre-aligned MIR image used as the fixed final reference grid.

        if_aligned
            Pre-aligned fluorescence image.

        mask_aligned
            IF-derived plaque mask that received exactly the same pre-alignment as IF.

        output_dir
            Folder where accepted NRRDs, JSON summary, and optional figures
            are written.

        sample_name
            Short sample identifier, for example "S1_N2".

        plaque_radius_um
            Plaque ROI radius in micrometers. Default 40 µm.

        plaque_zoom_count
            Number of plaque regions to show in the final zoom figure.
            Default 6. Use 0 to disable plaque zooms.

        plaque_zoom_selection
            Plaque selection strategy for the final zoom figure. Supported
            values are ``"largest"``, ``"random"``, and ``"mixed"``.
            ``"largest"`` reproduces the previous behavior. ``"random"``
            draws all displayed plaques randomly. ``"mixed"`` shows roughly
            half largest plaques and half random plaques. Default ``"largest"``.

        plaque_zoom_seed
            Seed for the local NumPy random generator used by random/mixed
            plaque QC sampling. Default 42. Use None for non-reproducible
            plaque selection. This seed does not affect registration.

        auto_refine
            Apply a small safe IF+mask rigid correction automatically when all
            quality gates pass. Default True.

        save_figures
            Save QC figures to output_dir/figures. Default True.

        show_figures
            Display figures interactively as well. Default False.

        gross_max_shift_um
            Largest BF-based residual translation tolerated before stopping
            and asking for better initial alignment. Default None means
            2 * plaque_radius_um.

        gross_max_rotation_deg
            Largest BF-based residual rotation tolerated. Default 1.0°.

        min_gradient_ncc
            Minimum structural Gradient NCC required. Default 0.10.

        min_edge_dice
            Minimum Edge Dice required. Default 0.40.

        direct_max_shift_um
            Largest direct MIR↔IF residual shift allowed for local refinement.
            Default None means plaque_radius_um.

        direct_max_rotation_deg
            Largest direct MIR↔IF residual rotation eligible for
            automatic rigid refinement. Default 0.5°.

        min_score_gain
            Minimum score improvement required before any automatic refinement.
            Default 0.003.

        rotation_min_extra_gain
            Minimum ADDITIONAL score improvement that rotation must provide
            beyond the best translation-only solution before rotation is
            applied. Default 0.003.

        post_max_residual_shift_um
            Maximum residual shift allowed after refinement. Default 10 µm.

        post_max_rotation_deg
            Maximum residual rotation allowed after refinement. Default 0.25°.

        post_max_score_gain
            Maximum remaining possible score gain after refinement.
            Default 0.003.
        """
        self.brightfield_source = brightfield
        self.mir_source = mir_aligned
        self.if_source = if_aligned
        self.mask_source = mask_aligned

        self.sample_name = str(sample_name)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"

        self.plaque_radius_um = float(plaque_radius_um)
        self.plaque_zoom_count = int(plaque_zoom_count)

        if self.plaque_zoom_count < 0:
            raise ValueError(
                "plaque_zoom_count must be >= 0."
            )

        self.plaque_zoom_selection = str(
            plaque_zoom_selection
        ).strip().lower()

        valid_plaque_zoom_selections = {
            "largest",
            "random",
            "mixed",
        }

        if (
            self.plaque_zoom_selection
            not in valid_plaque_zoom_selections
        ):
            raise ValueError(
                "plaque_zoom_selection must be one of: "
                "'largest', 'random', or 'mixed'."
            )

        self.plaque_zoom_seed = (
            None
            if plaque_zoom_seed is None
            else int(plaque_zoom_seed)
        )

        # Filled when final plaque QC is generated. Keeping the selection
        # metadata allows the exact QC sample to be reconstructed later.
        self.plaque_zoom_qc = None

        self.auto_refine = bool(auto_refine)
        self.auto_refine = self.auto_refine
        self.save_figures = bool(save_figures)
        self.show_figures = bool(show_figures)

        # -------------------------------------------------------------
        # Internal search resolution.
        # These reproduce the validated notebook and are intentionally
        # not routine public controls.
        # -------------------------------------------------------------
        self.coarse_target_um = 20.0
        self.coarse_max_shift_um = 80.0
        self.coarse_angles_deg = np.arange(
            -2.0,
            2.0001,
            0.5,
        )

        self.fine_target_um = 10.0
        self.fine_shift_radius_um = 20.0
        self.fine_angle_radius_deg = 0.5
        self.fine_angle_step_deg = 0.25
        self.support_erosion_um = 60.0

        # -------------------------------------------------------------
        # User-facing QC thresholds.
        # -------------------------------------------------------------
        self.gross_max_shift_um = (
            2.0 * self.plaque_radius_um
            if gross_max_shift_um is None
            else float(gross_max_shift_um)
        )

        self.gross_max_rot_deg = float(
            gross_max_rotation_deg
        )

        self.gross_min_grad_ncc = float(
            min_gradient_ncc
        )
        self.gross_min_edge_dice = float(
            min_edge_dice
        )

        self.direct_min_grad_ncc = float(
            min_gradient_ncc
        )
        self.direct_min_edge_dice = float(
            min_edge_dice
        )

        self.direct_max_shift_um = (
            self.plaque_radius_um
            if direct_max_shift_um is None
            else float(direct_max_shift_um)
        )

        self.direct_max_rot_deg = float(
            direct_max_rotation_deg
        )

        self.min_score_gain_for_autocorrection = float(
            min_score_gain
        )

        self.rotation_min_extra_gain = float(
            rotation_min_extra_gain
        )

        self.post_max_residual_shift_um = float(
            post_max_residual_shift_um
        )
        self.post_max_rot_deg = float(
            post_max_rotation_deg
        )
        self.post_max_score_gain = float(
            post_max_score_gain
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.save_figures:
            self.figures_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.if_refined_path = (
            self.output_dir
            / f"{self.sample_name}_IF_refined_to_MIR.nrrd"
        )

        self.mask_refined_path = (
            self.output_dir
            / f"{self.sample_name}_IF_mask_refined_to_MIR.nrrd"
        )

        self.if_on_mir_path = (
            self.output_dir
            / f"{self.sample_name}_IF_refined_on_MIR_grid.nrrd"
        )

        self.mask_on_mir_path = (
            self.output_dir
            / f"{self.sample_name}_IF_mask_refined_on_MIR_grid.nrrd"
        )

        self.summary_json_path = (
            self.output_dir
            / f"{self.sample_name}_MIR_IF_alignment_QC_summary.json"
        )

        self.saved_figures = []
        self._figure_counter = 0

        self.summary = None
        self.pipeline_status = None
        self.current_stage = "initialized"
        self.failure_reason = None

        self.if_refined_image = None
        self.mask_refined_image = None
        self.if_on_mir_image = None
        self.mask_on_mir_image = None

    def Run(self) -> Dict:
        """
        Run the complete MIR/IF alignment QC and optional residual refinement.

        Workflow
        --------
        1. Load BF, MIR, IF and IF-derived mask.
        2. Verify that IF and mask still share exactly the same geometry.
        3. Visually/structurally assess BF↔MIR and BF↔IF.
        4. Stop if the initial alignment is outside the local-refinement regime.
        5. Estimate the direct MIR↔IF residual.
        6. Compare translation-only with rigid refinement and apply rotation only when it adds sufficient evidence.
        7. Verify that IF/mask pixel arrays themselves were not altered.
        8. Sample accepted IF and mask onto the exact MIR pixel grid.
        9. Re-run MIR↔IF QC after correction.
        10. Save accepted NRRDs, figures, and one JSON QC summary.

        Returns
        -------
        dict
            The same summary that is written to
            <sample>_MIR_IF_alignment_QC_summary.json.

        Raises
        ------
        RuntimeError
            If a fail-fast gate rejects the registration. A partial JSON
            summary is still written before the error is raised.
        """
        try:
            self.current_stage = "load_inputs"
            self._load_inputs()

            self.current_stage = "if_mask_preflight"
            self._preflight_if_mask()

            self.current_stage = "bf_visual_validation"
            self._prepare_bf_views()
            self._plot_step1_validation()

            self.current_stage = "bf_registration_qc"
            self._run_bf_qc()
            self._apply_gross_alignment_gate()
            self._plot_bf_qc_diagnostics()

            self.current_stage = "direct_mir_if_qc"
            self._run_direct_mir_if_qc()
            self._apply_direct_gate()
            self._plot_direct_qc_diagnostics()

            self.current_stage = "candidate_refinement"
            self._prepare_candidate_correction()
            self._verify_pixels_unchanged()

            self.current_stage = "transfer_to_mir_grid"
            self._put_candidate_on_mir_grid()
            self._plot_before_after()

            self.current_stage = "post_refinement_qc"
            self._run_post_correction_qc()
            self._apply_post_correction_gate()

            self.current_stage = "save_accepted_images"
            self._write_accepted_images()

            self.current_stage = "final_visual_validation"
            self._prepare_final_mask()
            self._plot_final_mask()
            self._plot_final_six_panel()
            self._plot_plaque_zooms()

            self.current_stage = "complete"
            self._build_and_write_summary()

            return self.GetSummary()

        except Exception as exc:
            self.failure_reason = str(exc)

            if self.pipeline_status is None:
                self.pipeline_status = "STOPPED"

            self._write_json_checkpoint(
                stage=self.current_stage,
                run_status=self.pipeline_status,
                note=self.failure_reason,
            )

            raise

    def GetSummary(self) -> Dict:
        """Return the final summary dictionary."""
        if self.summary is None:
            raise RuntimeError(
                "Run() must complete successfully first."
            )

        return dict(
            self.summary
        )

    def GetFinalMaskImage(
        self,
    ) -> sitk.Image:
        """Return final plaque mask on the exact MIR grid."""
        if self.mask_on_mir_image is None:
            raise RuntimeError(
                "Run() must complete successfully first."
            )

        return sitk.Image(
            self.mask_on_mir_image
        )

    def GetFinalIFImage(
        self,
    ) -> sitk.Image:
        """Return final IF sampled onto the exact MIR grid."""
        if self.if_on_mir_image is None:
            raise RuntimeError(
                "Run() must complete successfully first."
            )

        return sitk.Image(
            self.if_on_mir_image
        )

    def GetOutputDirectory(
        self,
    ) -> Path:
        return self.output_dir

    def GetSavedFigures(self):
        return [
            Path(path)
            for path in self.saved_figures
        ]


    @staticmethod
    def GetParameterHelp() -> Dict[str, str]:
        """
        Return short help text for the user-facing parameters.

        This is useful in notebooks:

            import json
            print(json.dumps(
                MIRIFAlignmentQC.GetParameterHelp(),
                indent=2,
            ))
        """
        return {
            "plaque_radius_um": (
                "Plaque ROI radius in micrometers. Default 40. "
                "Also sets default direct_max_shift_um."
            ),
            "plaque_zoom_count": (
                "How many final plaque regions to show. Default 6. "
                "Use 0 to disable plaque zooms."
            ),
            "plaque_zoom_selection": (
                "How plaque zooms are chosen: 'largest', 'random', or "
                "'mixed'. 'largest' preserves historical behavior; "
                "'mixed' combines largest and random plaques. Default "
                "'largest'."
            ),
            "plaque_zoom_seed": (
                "Seed used only for random/mixed plaque zoom QC sampling. "
                "The same integer reproduces the same plaque sample. "
                "Use None for a new random sample each run. Default 42."
            ),
            "auto_refine": (
                "Apply a safe small rigid correction to IF + mask automatically. "
                "MIR and BF never move. Default True."
            ),
            "save_figures": (
                "Save all QC/validation PNGs in output_dir/figures. "
                "Default True."
            ),
            "show_figures": (
                "Also display figures interactively. Default False."
            ),
            "gross_max_shift_um": (
                "Largest BF-based residual shift tolerated before stopping "
                "and asking for a better initial alignment. "
                "Default: 2 × plaque_radius_um."
            ),
            "gross_max_rotation_deg": (
                "Largest BF-based residual rotation tolerated by the gross "
                "alignment gate. Default 1.0°."
            ),
            "min_gradient_ncc": (
                "Minimum gradient-structure correlation required by QC gates. "
                "Default 0.10."
            ),
            "min_edge_dice": (
                "Minimum overlap of strong structural edges. Default 0.40."
            ),
            "direct_max_shift_um": (
                "Largest direct MIR↔IF residual shift eligible for local "
                "refinement. Default: plaque_radius_um."
            ),
            "direct_max_rotation_deg": (
                "Largest direct MIR↔IF rotation compatible with "
                "automatic rigid refinement. Default 0.5°."
            ),
            "min_score_gain": (
                "Minimum structural-score gain required before any automatic "
                "refinement. Default 0.003."
            ),
            "rotation_min_extra_gain": (
                "Rotation is applied only when the best rigid solution "
                "improves the score by at least this amount beyond the best "
                "translation-only solution. Default 0.003."
            ),
            "post_max_residual_shift_um": (
                "Maximum remaining shift after correction. Default 10 µm."
            ),
            "post_max_rotation_deg": (
                "Maximum remaining rotation after correction. Default 0.25°."
            ),
            "post_max_score_gain": (
                "Maximum remaining possible structural-score gain after "
                "correction. Default 0.003."
            ),
        }

    def GetParameters(self) -> Dict:
        """Return the user-facing parameter values used for this QC run."""
        return {
            "plaque_radius_um": self.plaque_radius_um,
            "plaque_zoom_count": self.plaque_zoom_count,
            "plaque_zoom_selection": self.plaque_zoom_selection,
            "plaque_zoom_seed": self.plaque_zoom_seed,
            "auto_refine": self.auto_refine,
            "save_figures": self.save_figures,
            "show_figures": self.show_figures,
            "gross_max_shift_um": self.gross_max_shift_um,
            "gross_max_rotation_deg": self.gross_max_rot_deg,
            "min_gradient_ncc": self.direct_min_grad_ncc,
            "min_edge_dice": self.direct_min_edge_dice,
            "direct_max_shift_um": self.direct_max_shift_um,
            "direct_max_rotation_deg": self.direct_max_rot_deg,
            "min_score_gain": self.min_score_gain_for_autocorrection,
            "rotation_min_extra_gain": self.rotation_min_extra_gain,
            "post_max_residual_shift_um": self.post_max_residual_shift_um,
            "post_max_rotation_deg": self.post_max_rot_deg,
            "post_max_score_gain": self.post_max_score_gain,
        }

    # =================================================================
    # STEP 0 / 1 — input loading and pre-flight
    # =================================================================

    def _load_inputs(self):
        self.bf_img = _as_image(
            self.brightfield_source
        )

        self.mir_img = _as_image(
            self.mir_source
        )

        self.if_img = _as_image(
            self.if_source
        )

        self.mask_img = _as_image(
            self.mask_source
        )

        for name, image in {
            "Brightfield": self.bf_img,
            "MIR": self.mir_img,
            "IF": self.if_img,
            "Mask": self.mask_img,
        }.items():
            if (
                image.GetNumberOfComponentsPerPixel()
                != 1
            ):
                raise ValueError(
                    f"{name} must be a scalar image."
                )

            _squeeze2d(
                image
            )

        self._write_json_checkpoint(
            stage="inputs loaded",
            run_status="RUNNING",
        )

    def _preflight_if_mask(self):
        self.if_mask_same_grid = (
            _same_grid(
                self.if_img,
                self.mask_img,
            )
        )

        mask_arr = _squeeze2d(
            self.mask_img
        )

        self.mask_nonempty = bool(
            np.any(
                mask_arr > 0
            )
        )

        self._write_json_checkpoint(
            stage="IF-mask pre-flight",
            run_status=(
                "PASS"
                if (
                    self.if_mask_same_grid
                    and self.mask_nonempty
                )
                else "STOP"
            ),
            note=(
                None
                if (
                    self.if_mask_same_grid
                    and self.mask_nonempty
                )
                else (
                    "IF/mask geometry or mask-content "
                    "check failed."
                )
            ),
        )

        if not self.if_mask_same_grid:
            raise RuntimeError(
                "STOP — IF and plaque mask do not share "
                "identical geometry. They must receive the "
                "same registration transform before continuing."
            )

        if not self.mask_nonempty:
            raise RuntimeError(
                "STOP — plaque mask is empty."
            )

    # =================================================================
    # STEP 1 — BF-centered visual validation
    # =================================================================

    def _prepare_bf_views(self):
        self.mir_on_bf_img = (
            _resample_to_reference(
                self.mir_img,
                self.bf_img,
                sitk.sitkLinear,
                sitk.sitkFloat32,
                0,
            )
        )

        self.if_on_bf_img = (
            _resample_to_reference(
                self.if_img,
                self.bf_img,
                sitk.sitkLinear,
                sitk.sitkFloat32,
                0,
            )
        )

        self.mask_on_bf_img = (
            _resample_to_reference(
                self.mask_img,
                self.bf_img,
                sitk.sitkNearestNeighbor,
                sitk.sitkUInt8,
                0,
            )
        )

        self.bf = _squeeze2d(
            self.bf_img
        )

        self.mir_bf = _squeeze2d(
            self.mir_on_bf_img
        )

        self.if_bf = _squeeze2d(
            self.if_on_bf_img
        )

        self.mask_bf = (
            _squeeze2d(
                self.mask_on_bf_img
            )
            > 0
        )

        self.mir_support_bf = (
            np.isfinite(self.mir_bf)
            & (self.mir_bf != 0)
        )

        self.if_support_bf = (
            np.isfinite(self.if_bf)
            & (self.if_bf != 0)
        )

        self.bf_d = (
            _robust_normalize(
                self.bf
            )
        )

        self.mir_d = (
            _robust_normalize(
                self.mir_bf,
                valid=self.mir_support_bf,
            )
        )

        self.if_d = (
            _robust_normalize(
                self.if_bf,
                valid=self.if_support_bf,
            )
        )

        support_for_crop = (
            self.mir_support_bf
            | self.if_support_bf
        )

        rows, cols = np.nonzero(
            support_for_crop
        )

        if len(rows) == 0:
            raise RuntimeError(
                "No MIR/IF support found on the BF grid."
            )

        margin = 100

        y0 = max(
            0,
            int(rows.min()) - margin,
        )
        y1 = min(
            self.bf.shape[0],
            int(rows.max()) + margin + 1,
        )
        x0 = max(
            0,
            int(cols.min()) - margin,
        )
        x1 = min(
            self.bf.shape[1],
            int(cols.max()) + margin + 1,
        )

        self.bf_crop_d = (
            self.bf_d[
                y0:y1,
                x0:x1,
            ]
        )

        self.mir_crop_d = (
            self.mir_d[
                y0:y1,
                x0:x1,
            ]
        )

        self.if_crop_d = (
            self.if_d[
                y0:y1,
                x0:x1,
            ]
        )

        self.mask_crop = (
            self.mask_bf[
                y0:y1,
                x0:x1,
            ]
        )

    def _plot_step1_validation(self):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.imshow(
            self.bf_d,
            cmap="gray",
        )

        ax.imshow(
            self.mir_d,
            cmap="magma",
            alpha=0.40,
        )

        ax.set_title(
            "STEP 1 — Brightfield + aligned MIR"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_bf_plus_aligned_mir",
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.imshow(
            self.bf_d,
            cmap="gray",
        )

        ax.imshow(
            self.if_d,
            cmap="viridis",
            alpha=0.40,
        )

        ax.set_title(
            "STEP 1 — Brightfield + aligned IF"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_bf_plus_aligned_if",
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.imshow(
            self.if_d,
            cmap="gray",
        )

        if self.mask_bf.any():
            ax.contour(
                self.mask_bf,
                levels=[0.5],
                linewidths=0.8,
            )

        ax.set_title(
            "STEP 1 — aligned IF + plaque-mask contours"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_if_plus_mask",
        )

        for (
            first,
            second,
            cmap_second,
            title,
            filename,
        ) in [
            (
                self.bf_crop_d,
                self.mir_crop_d,
                "magma",
                "STEP 1 — ZOOM: Brightfield + MIR",
                "step1_zoom_bf_plus_mir",
            ),
            (
                self.bf_crop_d,
                self.if_crop_d,
                "viridis",
                "STEP 1 — ZOOM: Brightfield + IF",
                "step1_zoom_bf_plus_if",
            ),
        ]:
            fig, ax = plt.subplots(
                figsize=(9, 11)
            )

            ax.imshow(
                first,
                cmap="gray",
            )

            ax.imshow(
                second,
                cmap=cmap_second,
                alpha=0.40,
            )

            ax.set_title(
                title
            )
            ax.axis("off")

            self._finish_figure(
                fig,
                filename,
            )

        fig, ax = plt.subplots(
            figsize=(9, 11)
        )

        ax.imshow(
            self.if_crop_d,
            cmap="gray",
        )

        if self.mask_crop.any():
            ax.contour(
                self.mask_crop,
                levels=[0.5],
                linewidths=1.0,
            )

        ax.set_title(
            "STEP 1 — ZOOM: IF + plaque mask"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_zoom_if_plus_mask",
        )

        checker = _checkerboard(
            self.bf_crop_d,
            self.mir_crop_d,
            block=40,
        )

        fig, ax = plt.subplots(
            figsize=(9, 11)
        )

        ax.imshow(
            checker,
            cmap="gray",
        )

        ax.set_title(
            "STEP 1 — checkerboard: BF ↔ MIR"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_checkerboard_bf_mir",
        )

        checker = _checkerboard(
            self.bf_crop_d,
            self.if_crop_d,
            block=40,
        )

        fig, ax = plt.subplots(
            figsize=(9, 11)
        )

        ax.imshow(
            checker,
            cmap="gray",
        )

        ax.set_title(
            "STEP 1 — checkerboard: BF ↔ IF"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            "step1_checkerboard_bf_if",
        )

    # =================================================================
    # Structural residual search
    # =================================================================

    def _build_structural_maps(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        support: np.ndarray,
        spacing_xy_um: Tuple[
            float,
            float,
        ],
        erosion_um: float,
    ):
        support_inner = (
            _erode_support_physical(
                support,
                spacing_xy_um,
                erosion_um,
            )
        )

        _, fixed_grad = (
            _gradient_structure(
                fixed,
                valid=support_inner,
            )
        )

        _, moving_grad = (
            _gradient_structure(
                moving,
                valid=support_inner,
            )
        )

        fixed_edge = (
            _make_edge_map(
                fixed_grad,
                valid=support_inner,
            )
        )

        moving_edge = (
            _make_edge_map(
                moving_grad,
                valid=support_inner,
            )
        )

        return {
            "fixed_grad": fixed_grad,
            "moving_grad": moving_grad,
            "fixed_edge": fixed_edge,
            "moving_edge": moving_edge,
            "mask": support_inner,
        }

    def _search_transforms(
        self,
        maps: Dict,
        spacing_xy_um: Tuple[
            float,
            float,
        ],
        angles_deg: Iterable[
            float
        ],
        shift_x_um_values: Iterable[
            float
        ],
        shift_y_um_values: Iterable[
            float
        ],
    ):
        sx_um, sy_um = (
            spacing_xy_um
        )

        records = []

        for angle in angles_deg:
            moving_grad_rot = (
                ndi.rotate(
                    maps["moving_grad"],
                    angle=float(angle),
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
            )

            moving_edge_rot = (
                ndi.rotate(
                    maps["moving_edge"].astype(
                        np.uint8
                    ),
                    angle=float(angle),
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0,
                ).astype(bool)
            )

            moving_mask_rot = (
                ndi.rotate(
                    maps["mask"].astype(
                        np.uint8
                    ),
                    angle=float(angle),
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0,
                ).astype(bool)
            )

            for shift_y_um in (
                shift_y_um_values
            ):
                dy = int(
                    round(
                        shift_y_um
                        / sy_um
                    )
                )

                actual_y_um = (
                    dy * sy_um
                )

                for shift_x_um in (
                    shift_x_um_values
                ):
                    dx = int(
                        round(
                            shift_x_um
                            / sx_um
                        )
                    )

                    actual_x_um = (
                        dx * sx_um
                    )

                    result = (
                        _score_transform(
                            maps["fixed_grad"],
                            moving_grad_rot,
                            maps["fixed_edge"],
                            moving_edge_rot,
                            maps["mask"],
                            moving_mask_rot,
                            dy,
                            dx,
                        )
                    )

                    records.append({
                        "angle_deg": float(
                            angle
                        ),
                        "dx_px": int(dx),
                        "dy_px": int(dy),
                        "shift_x_um": float(
                            actual_x_um
                        ),
                        "shift_y_um": float(
                            actual_y_um
                        ),
                        **result,
                    })

        finite = [
            record
            for record in records
            if np.isfinite(
                record["score"]
            )
        ]

        if not finite:
            raise RuntimeError(
                "No finite structural registration "
                "scores were produced."
            )

        best = max(
            finite,
            key=lambda record: (
                record["score"]
            ),
        )

        return (
            records,
            best,
        )

    def _run_pair_qc(
        self,
        fixed_full: np.ndarray,
        moving_full: np.ndarray,
        moving_support_full: np.ndarray,
        spacing_xy_um: Tuple[
            float,
            float,
        ],
        label: str,
        rotation_search: bool = True,
    ):
        (
            fixed_crop,
            moving_crop,
            support_crop,
            bbox,
        ) = _crop_to_support(
            fixed_full,
            moving_full,
            moving_support_full,
            margin_px=10,
        )

        (
            fixed_coarse,
            moving_coarse,
            support_coarse,
            spacing_coarse,
        ) = _downsample_pair(
            fixed_crop,
            moving_crop,
            support_crop,
            spacing_xy_um,
            self.coarse_target_um,
        )

        maps_coarse = (
            self._build_structural_maps(
                fixed_coarse,
                moving_coarse,
                support_coarse,
                spacing_coarse,
                self.support_erosion_um,
            )
        )

        coarse_x = np.arange(
            -self.coarse_max_shift_um,
            (
                self.coarse_max_shift_um
                + 0.5 * spacing_coarse[0]
            ),
            spacing_coarse[0],
        )

        coarse_y = np.arange(
            -self.coarse_max_shift_um,
            (
                self.coarse_max_shift_um
                + 0.5 * spacing_coarse[1]
            ),
            spacing_coarse[1],
        )

        coarse_angles = (
            self.coarse_angles_deg
            if rotation_search
            else [0.0]
        )

        _, coarse_best = (
            self._search_transforms(
                maps_coarse,
                spacing_coarse,
                coarse_angles,
                coarse_x,
                coarse_y,
            )
        )

        (
            fixed_fine,
            moving_fine,
            support_fine,
            spacing_fine,
        ) = _downsample_pair(
            fixed_crop,
            moving_crop,
            support_crop,
            spacing_xy_um,
            self.fine_target_um,
        )

        maps_fine = (
            self._build_structural_maps(
                fixed_fine,
                moving_fine,
                support_fine,
                spacing_fine,
                self.support_erosion_um,
            )
        )

        if rotation_search:
            fine_angles = np.arange(
                (
                    coarse_best["angle_deg"]
                    - self.fine_angle_radius_deg
                ),
                (
                    coarse_best["angle_deg"]
                    + self.fine_angle_radius_deg
                    + 0.5
                    * self.fine_angle_step_deg
                ),
                self.fine_angle_step_deg,
            )
        else:
            fine_angles = [0.0]

        fine_x = np.arange(
            (
                coarse_best["shift_x_um"]
                - self.fine_shift_radius_um
            ),
            (
                coarse_best["shift_x_um"]
                + self.fine_shift_radius_um
                + 0.5 * spacing_fine[0]
            ),
            spacing_fine[0],
        )

        fine_y = np.arange(
            (
                coarse_best["shift_y_um"]
                - self.fine_shift_radius_um
            ),
            (
                coarse_best["shift_y_um"]
                + self.fine_shift_radius_um
                + 0.5 * spacing_fine[1]
            ),
            spacing_fine[1],
        )

        (
            fine_records,
            fine_best,
        ) = self._search_transforms(
            maps_fine,
            spacing_fine,
            fine_angles,
            fine_x,
            fine_y,
        )

        (
            zero_records,
            _,
        ) = self._search_transforms(
            maps_fine,
            spacing_fine,
            [0.0],
            [0.0],
            [0.0],
        )

        zero = zero_records[0]

        correction_magnitude_um = (
            float(
                np.hypot(
                    fine_best[
                        "shift_x_um"
                    ],
                    fine_best[
                        "shift_y_um"
                    ],
                )
            )
        )

        score_gain = float(
            fine_best["score"]
            - zero["score"]
        )

        return {
            "label": label,
            "bbox_xyxy": bbox,
            "coarse_best": coarse_best,
            "fine_best": fine_best,
            "zero": zero,
            "correction_magnitude_um": (
                correction_magnitude_um
            ),
            "score_gain_vs_zero": (
                score_gain
            ),
            "fine_records": fine_records,
            "spacing_fine_um": (
                spacing_fine
            ),
            "fixed_crop": fixed_crop,
            "moving_crop": moving_crop,
            "support_crop": support_crop,
            "rotation_search": bool(
                rotation_search
            ),
        }

    # =================================================================
    # STEP 2 — BF-based QC + gross alignment gate
    # =================================================================

    def _run_bf_qc(self):
        self.bf_spacing_xy_um = (
            float(
                self.bf_img.GetSpacing()[0]
            )
            * 1000.0,
            float(
                self.bf_img.GetSpacing()[1]
            )
            * 1000.0,
        )

        self.bf_mir_qc = (
            self._run_pair_qc(
                self.bf,
                self.mir_bf,
                self.mir_support_bf,
                self.bf_spacing_xy_um,
                "BF ↔ MIR",
            )
        )

        self.bf_if_qc = (
            self._run_pair_qc(
                self.bf,
                self.if_bf,
                self.if_support_bf,
                self.bf_spacing_xy_um,
                "BF ↔ IF",
            )
        )

        self.bf_mir_status = (
            self._pair_status(
                self.bf_mir_qc
            )
        )

        self.bf_if_status = (
            self._pair_status(
                self.bf_if_qc
            )
        )

        self._write_json_checkpoint(
            stage="BF residual estimation",
            run_status="RUNNING",
        )

    @staticmethod
    def _pair_status(
        qc: Dict,
    ) -> str:
        best = qc[
            "fine_best"
        ]

        shift_um = qc[
            "correction_magnitude_um"
        ]

        angle_deg = abs(
            best["angle_deg"]
        )

        gain = qc[
            "score_gain_vs_zero"
        ]

        if (
            shift_um <= 10.0
            and angle_deg <= 0.5
        ):
            return "PASS"

        if (
            shift_um <= 20.0
            and angle_deg <= 1.0
        ):
            return "INSPECT"

        if gain >= 0.02:
            return "REFINE"

        return "INSPECT"

    def _gross_alignment_checks(
        self,
        qc: Dict,
    ) -> Dict[
        str,
        bool,
    ]:
        return {
            "shift_ok": (
                qc[
                    "correction_magnitude_um"
                ]
                <= self.gross_max_shift_um
            ),
            "rotation_ok": (
                abs(
                    qc["fine_best"][
                        "angle_deg"
                    ]
                )
                <= self.gross_max_rot_deg
            ),
            "gradient_ncc_ok": (
                qc["zero"]["ncc"]
                >= self.gross_min_grad_ncc
            ),
            "edge_dice_ok": (
                qc["zero"]["edge_dice"]
                >= self.gross_min_edge_dice
            ),
        }

    def _apply_gross_alignment_gate(
        self,
    ):
        self.bf_mir_gross_checks = (
            self._gross_alignment_checks(
                self.bf_mir_qc
            )
        )

        self.bf_if_gross_checks = (
            self._gross_alignment_checks(
                self.bf_if_qc
            )
        )

        self.bf_mir_gross_ok = all(
            self.bf_mir_gross_checks.values()
        )

        self.bf_if_gross_ok = all(
            self.bf_if_gross_checks.values()
        )

        self.initial_alignment_ok = (
            self.bf_mir_gross_ok
            and self.bf_if_gross_ok
        )

        self._write_json_checkpoint(
            stage="gross alignment gate",
            run_status=(
                "PASS"
                if self.initial_alignment_ok
                else "STOP"
            ),
            note=(
                None
                if self.initial_alignment_ok
                else (
                    "Initial alignment is too poor "
                    "for local residual analysis. "
                    "Improve the initial alignment first."
                )
            ),
        )

        if (
            not self.initial_alignment_ok
        ):
            raise RuntimeError(
                "STOP — initial registration is too far "
                "from the local-refinement regime. "
                "Please perform/refine the initial registration "
                "in initial alignment first, export aligned MIR, IF and mask, "
                "then rerun MIRIFAlignmentQC. "
                "No final MIR-grid training mask was created."
            )

    # =================================================================
    # Diagnostic plots used by BF and direct MIR↔IF QC
    # =================================================================

    def _apply_best_to_moving_crop(
        self,
        moving_crop: np.ndarray,
        support_crop: np.ndarray,
        spacing_xy_um: Tuple[
            float,
            float,
        ],
        best: Dict,
    ):
        sx_um, sy_um = (
            spacing_xy_um
        )

        rotated = ndi.rotate(
            moving_crop,
            angle=best["angle_deg"],
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )

        rotated_support = (
            ndi.rotate(
                support_crop.astype(
                    np.uint8
                ),
                angle=best["angle_deg"],
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            ).astype(bool)
        )

        dy = (
            best["shift_y_um"]
            / sy_um
        )

        dx = (
            best["shift_x_um"]
            / sx_um
        )

        shifted = ndi.shift(
            rotated,
            shift=(dy, dx),
            order=1,
            mode="constant",
            cval=0.0,
        )

        shifted_support = (
            ndi.shift(
                rotated_support.astype(
                    np.uint8
                ),
                shift=(dy, dx),
                order=0,
                mode="constant",
                cval=0,
            ).astype(bool)
        )

        return (
            shifted,
            shifted_support,
        )

    def _show_current_vs_best(
        self,
        qc: Dict,
        spacing_xy_um: Tuple[
            float,
            float,
        ],
        prefix: str,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        fixed = qc[
            "fixed_crop"
        ]

        moving = qc[
            "moving_crop"
        ]

        support = qc[
            "support_crop"
        ]

        (
            best_moving,
            best_support,
        ) = (
            self._apply_best_to_moving_crop(
                moving,
                support,
                spacing_xy_um,
                qc["fine_best"],
            )
        )

        fixed_display = (
            _robust_normalize(
                fixed,
                valid=np.isfinite(
                    fixed
                ),
            )
        )

        moving_display = (
            _robust_normalize(
                moving,
                valid=support,
            )
        )

        best_display = (
            _robust_normalize(
                best_moving,
                valid=best_support,
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 11)
        )

        ax.imshow(
            fixed_display,
            cmap="gray",
        )

        ax.imshow(
            moving_display,
            cmap="magma",
            alpha=0.38,
        )

        ax.set_title(
            qc["label"]
            + "\nCurrent placement"
        )
        ax.axis("off")

        self._finish_figure(
            fig,
            f"{prefix}_current",
        )

        fig, ax = plt.subplots(
            figsize=(9, 11)
        )

        ax.imshow(
            fixed_display,
            cmap="gray",
        )

        ax.imshow(
            best_display,
            cmap="magma",
            alpha=0.38,
        )

        best = qc[
            "fine_best"
        ]

        ax.set_title(
            qc["label"]
            + "\nBest nearby structural correction"
            + (
                f"\nΔx={best['shift_x_um']:.1f} µm, "
                f"Δy={best['shift_y_um']:.1f} µm, "
                f"rot={best['angle_deg']:.2f}°"
            )
        )

        ax.axis("off")

        self._finish_figure(
            fig,
            f"{prefix}_best",
        )

    def _plot_translation_heatmap(
        self,
        qc: Dict,
        name: str,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        records = qc[
            "fine_records"
        ]

        best_angle = qc[
            "fine_best"
        ]["angle_deg"]

        same_angle = [
            record
            for record in records
            if (
                np.isclose(
                    record["angle_deg"],
                    best_angle,
                )
                and np.isfinite(
                    record["score"]
                )
            )
        ]

        xs = sorted(
            set(
                record["shift_x_um"]
                for record in same_angle
            )
        )

        ys = sorted(
            set(
                record["shift_y_um"]
                for record in same_angle
            )
        )

        grid = np.full(
            (
                len(ys),
                len(xs),
            ),
            np.nan,
        )

        x_index = {
            value: index
            for index, value in enumerate(xs)
        }

        y_index = {
            value: index
            for index, value in enumerate(ys)
        }

        for record in same_angle:
            grid[
                y_index[
                    record["shift_y_um"]
                ],
                x_index[
                    record["shift_x_um"]
                ],
            ] = record["score"]

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[
                min(xs),
                max(xs),
                min(ys),
                max(ys),
            ],
        )

        fig.colorbar(
            image,
            ax=ax,
            label=(
                "structural similarity score"
            ),
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.scatter(
            [
                qc["fine_best"][
                    "shift_x_um"
                ]
            ],
            [
                qc["fine_best"][
                    "shift_y_um"
                ]
            ],
            marker="x",
            s=80,
        )

        ax.set_xlabel(
            "Residual X correction (µm)"
        )

        ax.set_ylabel(
            "Residual Y correction (µm)"
        )

        ax.set_title(
            qc["label"]
            + (
                "\nTranslation score at rotation "
                f"{best_angle:.2f}°"
            )
        )

        self._finish_figure(
            fig,
            name,
        )

    def _plot_rotation_profile(
        self,
        qc: Dict,
        name: str,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        records = [
            record
            for record in qc[
                "fine_records"
            ]
            if np.isfinite(
                record["score"]
            )
        ]

        angles = sorted(
            set(
                record["angle_deg"]
                for record in records
            )
        )

        best_scores = []

        for angle in angles:
            candidates = [
                record
                for record in records
                if np.isclose(
                    record["angle_deg"],
                    angle,
                )
            ]

            best = max(
                candidates,
                key=lambda record: (
                    record["score"]
                ),
            )

            best_scores.append(
                best["score"]
            )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            angles,
            best_scores,
            marker="o",
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.axvline(
            qc["fine_best"][
                "angle_deg"
            ],
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel(
            "Residual rotation (degrees)"
        )

        ax.set_ylabel(
            "Best structural score"
        )

        ax.set_title(
            qc["label"]
            + "\nBest score at each tested rotation"
        )

        self._finish_figure(
            fig,
            name,
        )

    def _plot_bf_qc_diagnostics(
        self,
    ):
        self._show_current_vs_best(
            self.bf_mir_qc,
            self.bf_spacing_xy_um,
            "bf_mir",
        )

        self._show_current_vs_best(
            self.bf_if_qc,
            self.bf_spacing_xy_um,
            "bf_if",
        )

        self._plot_translation_heatmap(
            self.bf_mir_qc,
            "bf_mir_translation_heatmap",
        )

        self._plot_translation_heatmap(
            self.bf_if_qc,
            "bf_if_translation_heatmap",
        )

        self._plot_rotation_profile(
            self.bf_mir_qc,
            "bf_mir_rotation_profile",
        )

        self._plot_rotation_profile(
            self.bf_if_qc,
            "bf_if_rotation_profile",
        )

    # =================================================================
    # STEP 3 — direct MIR↔IF residual estimation + gate
    # =================================================================

    def _run_direct_mir_if_qc(
        self,
    ):
        """
        Compare MIR and IF directly and decide whether rotation is actually needed.

        Two optimizations are run on the same MIR/IF pair:

        1. translation-only:
              dx, dy, angle = 0

        2. rigid:
              dx, dy, angle

        Rotation is selected only when the rigid solution improves the score
        over the best translation-only solution by at least
        ``rotation_min_extra_gain``.

        This prevents a small non-zero angle from being applied merely because
        it is the numerical optimum when translation alone is essentially just
        as good.
        """
        self.mir_native = (
            _squeeze2d(
                self.mir_img
            )
        )

        self.if_on_mir_for_qc_img = (
            _resample_to_reference(
                self.if_img,
                self.mir_img,
                sitk.sitkLinear,
                sitk.sitkFloat32,
                0,
            )
        )

        self.if_on_mir_for_qc = (
            _squeeze2d(
                self.if_on_mir_for_qc_img
            )
        )

        self.if_on_mir_support = (
            np.isfinite(
                self.if_on_mir_for_qc
            )
            & (
                self.if_on_mir_for_qc
                != 0
            )
        )

        self.mir_spacing_xy_um = (
            float(
                self.mir_img.GetSpacing()[0]
            )
            * 1000.0,
            float(
                self.mir_img.GetSpacing()[1]
            )
            * 1000.0,
        )

        # -------------------------------------------------------------
        # Full small-rigid search: translation + rotation.
        # -------------------------------------------------------------
        self.mir_if_direct_qc = (
            self._run_pair_qc(
                self.mir_native,
                self.if_on_mir_for_qc,
                self.if_on_mir_support,
                self.mir_spacing_xy_um,
                "MIR ↔ IF direct rigid",
                rotation_search=True,
            )
        )

        # -------------------------------------------------------------
        # Independent translation-only search on the same data.
        # -------------------------------------------------------------
        self.mir_if_translation_only_qc = (
            self._run_pair_qc(
                self.mir_native,
                self.if_on_mir_for_qc,
                self.if_on_mir_support,
                self.mir_spacing_xy_um,
                "MIR ↔ IF translation only",
                rotation_search=False,
            )
        )

        rigid_best = (
            self.mir_if_direct_qc[
                "fine_best"
            ]
        )

        translation_best = (
            self.mir_if_translation_only_qc[
                "fine_best"
            ]
        )

        rigid_score = float(
            rigid_best["score"]
        )

        translation_score = float(
            translation_best["score"]
        )

        self.rotation_extra_gain = float(
            rigid_score
            - translation_score
        )

        # Numerical roundoff can make the nominally more flexible rigid
        # solution lower by a tiny amount.  Such a result never justifies
        # rotation.
        self.rotation_required = bool(
            abs(
                float(
                    rigid_best[
                        "angle_deg"
                    ]
                )
            )
            > 1e-12
            and self.rotation_extra_gain
            >= self.rotation_min_extra_gain
        )

        if self.rotation_required:
            self.selected_refinement_mode = (
                "rigid"
            )
            selected_qc = (
                self.mir_if_direct_qc
            )
            selected_best = rigid_best
        else:
            self.selected_refinement_mode = (
                "translation_only"
            )
            selected_qc = (
                self.mir_if_translation_only_qc
            )
            selected_best = (
                translation_best
            )

        self.selected_direct_qc = (
            selected_qc
        )

        self.direct_dx_um = float(
            selected_best[
                "shift_x_um"
            ]
        )

        self.direct_dy_um = float(
            selected_best[
                "shift_y_um"
            ]
        )

        self.direct_rot_deg = float(
            selected_best[
                "angle_deg"
            ]
        )

        self.direct_shift_mag_um = float(
            selected_qc[
                "correction_magnitude_um"
            ]
        )

        self.direct_score_gain = float(
            selected_qc[
                "score_gain_vs_zero"
            ]
        )

        # Useful diagnostics retained separately for JSON reporting.
        self.rigid_candidate_dx_um = float(
            rigid_best[
                "shift_x_um"
            ]
        )
        self.rigid_candidate_dy_um = float(
            rigid_best[
                "shift_y_um"
            ]
        )
        self.rigid_candidate_rot_deg = float(
            rigid_best[
                "angle_deg"
            ]
        )

        self.translation_candidate_dx_um = float(
            translation_best[
                "shift_x_um"
            ]
        )
        self.translation_candidate_dy_um = float(
            translation_best[
                "shift_y_um"
            ]
        )

        self._write_json_checkpoint(
            stage=(
                "direct MIR-IF residual estimate"
            ),
            run_status="RUNNING",
        )


    @staticmethod
    def _search_boundary_flags(
        qc: Dict,
    ) -> Dict[
        str,
        bool,
    ]:
        records = [
            record
            for record in qc[
                "fine_records"
            ]
            if np.isfinite(
                record["score"]
            )
        ]

        best = qc[
            "fine_best"
        ]

        xs = np.asarray(
            sorted(
                set(
                    record["shift_x_um"]
                    for record in records
                )
            ),
            dtype=float,
        )

        ys = np.asarray(
            sorted(
                set(
                    record["shift_y_um"]
                    for record in records
                )
            ),
            dtype=float,
        )

        angles = np.asarray(
            sorted(
                set(
                    record["angle_deg"]
                    for record in records
                )
            ),
            dtype=float,
        )

        return {
            "x_boundary": bool(
                np.isclose(
                    best["shift_x_um"],
                    xs.min(),
                )
                or np.isclose(
                    best["shift_x_um"],
                    xs.max(),
                )
            ),
            "y_boundary": bool(
                np.isclose(
                    best["shift_y_um"],
                    ys.min(),
                )
                or np.isclose(
                    best["shift_y_um"],
                    ys.max(),
                )
            ),
            "angle_boundary": bool(
                np.isclose(
                    best["angle_deg"],
                    angles.min(),
                )
                or np.isclose(
                    best["angle_deg"],
                    angles.max(),
                )
            ),
        }

    def _apply_direct_gate(
        self,
    ):
        """
        Apply strict trust rules to the selected direct refinement.

        When rotation does not add enough score beyond translation alone,
        the selected correction is translation-only and the applied angle is
        exactly 0 degrees.

        When rotation is demonstrably useful, the selected correction is the
        rigid solution and the usual direct rotation limit applies.
        """
        if self.rotation_required:
            selected_boundary = (
                self._search_boundary_flags(
                    self.mir_if_direct_qc
                )
            )
        else:
            # Translation-only search has only angle=0, so its generic
            # boundary detector would always label angle as a boundary.
            # Only x/y boundaries matter when no rotation will be applied.
            translation_boundary = (
                self._search_boundary_flags(
                    self.mir_if_translation_only_qc
                )
            )

            selected_boundary = {
                "x_boundary": bool(
                    translation_boundary[
                        "x_boundary"
                    ]
                ),
                "y_boundary": bool(
                    translation_boundary[
                        "y_boundary"
                    ]
                ),
                "angle_boundary": False,
            }

        self.direct_boundary = (
            selected_boundary
        )

        self.direct_search_hit_boundary = (
            any(
                self.direct_boundary.values()
            )
        )

        # Current/no-correction structural evidence is common to both searches.
        zero = (
            self.mir_if_direct_qc[
                "zero"
            ]
        )

        self.direct_structure_ok = (
            zero["ncc"]
            >= self.direct_min_grad_ncc
            and zero["edge_dice"]
            >= self.direct_min_edge_dice
        )

        self.direct_local_error_ok = (
            self.direct_shift_mag_um
            <= self.direct_max_shift_um
            and abs(
                self.direct_rot_deg
            )
            <= self.direct_max_rot_deg
            and not self.direct_search_hit_boundary
        )

        self.direct_match_trustworthy = (
            self.direct_structure_ok
            and self.direct_local_error_ok
        )

        self.correction_recommended = (
            self.direct_match_trustworthy
            and (
                self.direct_shift_mag_um
                > 0.0
                or abs(
                    self.direct_rot_deg
                )
                > 1e-12
            )
            and self.direct_score_gain
            >= self.min_score_gain_for_autocorrection
        )

        self._write_json_checkpoint(
            stage=(
                "direct MIR-IF evidence gate"
            ),
            run_status=(
                "PASS"
                if self.direct_match_trustworthy
                else "STOP"
            ),
            note=(
                None
                if self.direct_match_trustworthy
                else (
                    "Direct MIR-IF agreement is not "
                    "trustworthy enough for automatic refinement."
                )
            ),
        )

        if not (
            self.direct_match_trustworthy
        ):
            raise RuntimeError(
                "STOP — direct MIR↔IF structural agreement "
                "is not strong enough for automatic refinement. "
                "Do not generate a training mask from this run. "
                "Improve the external initial alignment first."
            )

        self.rigid_refinement_safe = (
            self.direct_match_trustworthy
            and self.direct_shift_mag_um
            <= self.direct_max_shift_um
            and abs(
                self.direct_rot_deg
            )
            <= self.direct_max_rot_deg
            and not self.direct_search_hit_boundary
        )


    def _plot_direct_qc_diagnostics(
        self,
    ):
        # Translation-only heatmap shows the best shift without allowing
        # rotation to "help" the translation estimate.
        self._plot_translation_heatmap(
            self.mir_if_translation_only_qc,
            "direct_mir_if_translation_only_heatmap",
        )

        # Rigid rotation profile shows whether an angle actually improves
        # the score beyond translation alone.
        self._plot_rotation_profile(
            self.mir_if_direct_qc,
            "direct_mir_if_rotation_profile",
        )

        # Show the correction that was actually selected by the necessity rule.
        self._show_current_vs_best(
            self.selected_direct_qc,
            self.mir_spacing_xy_um,
            "direct_mir_if_selected",
        )


    # =================================================================
    # STEP 4 — candidate rigid correction of IF + mask together
    # =================================================================

    def _direct_refinement_center_physical(
        self,
    ) -> Tuple[float, ...]:
        """
        Return the physical rotation center corresponding to the direct-QC crop.

        The direct MIR↔IF search rotates the cropped moving array about the
        crop center.  Using this same physical center for the geometry update
        keeps the estimated and applied rigid corrections consistent.
        """
        (
            x0,
            x1,
            y0,
            y1,
        ) = self.mir_if_direct_qc[
            "bbox_xyxy"
        ]

        center_index = [
            0.5 * (
                float(x0)
                + float(x1 - 1)
            ),
            0.5 * (
                float(y0)
                + float(y1 - 1)
            ),
        ]

        if (
            self.mir_img.GetDimension()
            == 3
        ):
            center_index.append(
                0.0
            )

        return tuple(
            float(v)
            for v in self.mir_img.TransformContinuousIndexToPhysicalPoint(
                tuple(center_index)
            )
        )

    def _prepare_candidate_correction(
        self,
    ):
        """
        Prepare the candidate small rigid refinement.

        MIR remains fixed.

        IF and the plaque mask receive the same selected correction:
            - x translation,
            - y translation,
            - in-plane rotation only when rotation adds enough evidence.

        Only image geometry (Origin + Direction) is changed here, so the
        original IF and mask pixel arrays remain exactly unchanged.
        """
        self.correction_applied = (
            self.auto_refine
            and self.rigid_refinement_safe
            and self.correction_recommended
        )

        if self.correction_applied:
            self.refinement_center_physical = (
                self._direct_refinement_center_physical()
            )

            self.if_refined_image = (
                _rigid_adjust_image_geometry(
                    self.if_img,
                    reference=self.mir_img,
                    dx_um=self.direct_dx_um,
                    dy_um=self.direct_dy_um,
                    angle_deg=self.direct_rot_deg,
                    center_physical=(
                        self.refinement_center_physical
                    ),
                )
            )

            self.mask_refined_image = (
                _rigid_adjust_image_geometry(
                    self.mask_img,
                    reference=self.mir_img,
                    dx_um=self.direct_dx_um,
                    dy_um=self.direct_dy_um,
                    angle_deg=self.direct_rot_deg,
                    center_physical=(
                        self.refinement_center_physical
                    ),
                )
            )

        else:
            self.refinement_center_physical = None

            self.if_refined_image = (
                sitk.Image(
                    self.if_img
                )
            )

            self.mask_refined_image = (
                sitk.Image(
                    self.mask_img
                )
            )

        self._write_json_checkpoint(
            stage=(
                "candidate rigid correction prepared"
            ),
            run_status="RUNNING",
        )

    def _verify_pixels_unchanged(
        self,
    ):
        self.if_pixels_unchanged = (
            np.array_equal(
                sitk.GetArrayFromImage(
                    self.if_img
                ),
                sitk.GetArrayFromImage(
                    self.if_refined_image
                ),
            )
        )

        self.mask_pixels_unchanged = (
            np.array_equal(
                sitk.GetArrayFromImage(
                    self.mask_img
                ),
                sitk.GetArrayFromImage(
                    self.mask_refined_image
                ),
            )
        )

        if (
            not self.if_pixels_unchanged
            or not self.mask_pixels_unchanged
        ):
            self._write_json_checkpoint(
                stage=(
                    "pixel-preservation check"
                ),
                run_status="STOP",
                note=(
                    "Geometry-only refinement unexpectedly "
                    "changed IF or mask pixel values."
                ),
            )

            raise RuntimeError(
                "STOP — geometry-only refinement "
                "changed IF or mask pixels."
            )

        self._write_json_checkpoint(
            stage="pixel-preservation check",
            run_status="PASS",
        )

    # =================================================================
    # STEP 5 — sample accepted candidate onto MIR grid
    # =================================================================

    def _put_candidate_on_mir_grid(
        self,
    ):
        self.if_on_mir_image = (
            _resample_to_reference(
                self.if_refined_image,
                self.mir_img,
                sitk.sitkLinear,
                sitk.sitkFloat32,
                0,
            )
        )

        self.mask_on_mir_image = (
            _resample_to_reference(
                self.mask_refined_image,
                self.mir_img,
                sitk.sitkNearestNeighbor,
                sitk.sitkUInt8,
                0,
            )
        )

        self.if_on_mir_grid = (
            _same_grid(
                self.if_on_mir_image,
                self.mir_img,
            )
        )

        self.mask_on_mir_grid = (
            _same_grid(
                self.mask_on_mir_image,
                self.mir_img,
            )
        )

        self._write_json_checkpoint(
            stage=(
                "candidate common-MIR-grid check"
            ),
            run_status="RUNNING",
        )

    def _plot_before_after(
        self,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        mir_show = (
            _robust_normalize(
                self.mir_native
            )
        )

        if_before_show = (
            _robust_normalize(
                self.if_on_mir_for_qc,
                valid=self.if_on_mir_support,
            )
        )

        if_after = (
            _squeeze2d(
                self.if_on_mir_image
            )
        )

        if_after_support = (
            np.isfinite(if_after)
            & (if_after != 0)
        )

        if_after_show = (
            _robust_normalize(
                if_after,
                valid=if_after_support,
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 12)
        )

        ax.imshow(
            mir_show,
            cmap="gray",
        )

        ax.imshow(
            if_before_show,
            cmap="magma",
            alpha=0.38,
        )

        ax.set_title(
            "MIR + IF BEFORE automatic direct refinement"
        )

        ax.axis("off")

        self._finish_figure(
            fig,
            "mir_if_before_refinement",
        )

        fig, ax = plt.subplots(
            figsize=(9, 12)
        )

        ax.imshow(
            mir_show,
            cmap="gray",
        )

        ax.imshow(
            if_after_show,
            cmap="magma",
            alpha=0.38,
        )

        if self.correction_applied:
            ax.set_title(
                "MIR + IF AFTER automatic direct refinement"
                + (
                    f"\nΔx={self.direct_dx_um:.1f} µm, "
                    f"Δy={self.direct_dy_um:.1f} µm, "
                    f"θ={self.direct_rot_deg:.2f}°"
                )
            )
        else:
            ax.set_title(
                "MIR + IF — no automatic correction applied"
            )

        ax.axis("off")

        self._finish_figure(
            fig,
            "mir_if_after_refinement",
        )

    # =================================================================
    # STEP 6 — closed-loop verification
    # =================================================================

    def _run_post_correction_qc(
        self,
    ):
        if_after_for_qc = (
            _squeeze2d(
                self.if_on_mir_image
            )
        )

        if_after_support = (
            np.isfinite(
                if_after_for_qc
            )
            & (
                if_after_for_qc
                != 0
            )
        )

        self.mir_if_post_qc = (
            self._run_pair_qc(
                self.mir_native,
                if_after_for_qc,
                if_after_support,
                self.mir_spacing_xy_um,
                (
                    "MIR ↔ IF "
                    "after refinement"
                ),
            )
        )

        self.pre_score = float(
            self.mir_if_direct_qc[
                "zero"
            ]["score"]
        )

        self.post_score = float(
            self.mir_if_post_qc[
                "zero"
            ]["score"]
        )

        self.post_best_shift_um = (
            float(
                self.mir_if_post_qc[
                    "correction_magnitude_um"
                ]
            )
        )

        self.post_best_rot_deg = (
            float(
                self.mir_if_post_qc[
                    "fine_best"
                ]["angle_deg"]
            )
        )

        self.post_gain = float(
            self.mir_if_post_qc[
                "score_gain_vs_zero"
            ]
        )

        self.score_not_worse = (
            self.post_score
            >= self.pre_score
            - 1e-12
        )

        self._write_json_checkpoint(
            stage=(
                "post-correction direct MIR-IF QC"
            ),
            run_status="RUNNING",
        )

    def _apply_post_correction_gate(
        self,
    ):
        if self.correction_applied:
            self.correction_accepted = (
                self.score_not_worse
                and self.post_best_shift_um
                <= self.post_max_residual_shift_um
                and abs(
                    self.post_best_rot_deg
                )
                <= self.post_max_rot_deg
                and self.post_gain
                <= self.post_max_score_gain
            )

            if not (
                self.correction_accepted
            ):
                self._write_json_checkpoint(
                    stage=(
                        "post-correction acceptance gate"
                    ),
                    run_status="STOP",
                    note=(
                        "Candidate automatic correction failed "
                        "closed-loop validation."
                    ),
                )

                raise RuntimeError(
                    "STOP — automatic refinement was unstable "
                    "or failed closed-loop validation. "
                    "No refined outputs and no MIR-grid training "
                    "mask were saved. Return to initial alignment."
                )

        else:
            self.correction_accepted = (
                self.direct_match_trustworthy
                and self.direct_shift_mag_um
                <= self.post_max_residual_shift_um
                and abs(
                    self.direct_rot_deg
                )
                <= self.post_max_rot_deg
            )

            if not (
                self.correction_accepted
            ):
                self._write_json_checkpoint(
                    stage=(
                        "post-correction retention gate"
                    ),
                    run_status="STOP",
                    note=(
                        "Current initial alignment placement cannot be "
                        "retained without refinement."
                    ),
                )

                raise RuntimeError(
                    "STOP — current initial alignment placement is not "
                    "close enough to accept without refinement. "
                    "No MIR-grid training mask was saved."
                )

        self._write_json_checkpoint(
            stage=(
                "post-correction acceptance gate"
            ),
            run_status="PASS",
        )

    # =================================================================
    # STEP 7 — accepted outputs + final visual validation
    # =================================================================

    def _write_accepted_images(
        self,
    ):
        if self.correction_applied:
            sitk.WriteImage(
                self.if_refined_image,
                str(
                    self.if_refined_path
                ),
            )

            sitk.WriteImage(
                self.mask_refined_image,
                str(
                    self.mask_refined_path
                ),
            )

        sitk.WriteImage(
            self.if_on_mir_image,
            str(
                self.if_on_mir_path
            ),
        )

        sitk.WriteImage(
            self.mask_on_mir_image,
            str(
                self.mask_on_mir_path
            ),
        )

        self._write_json_checkpoint(
            stage=(
                "accepted outputs written to disk"
            ),
            run_status="PASS",
        )

    def _prepare_final_mask(
        self,
    ):
        self.mask_final = (
            _squeeze2d(
                self.mask_on_mir_image
            )
            > 0
        )

        self.plaque_positive_mir_pixels = (
            int(
                self.mask_final.sum()
            )
        )

        self.mir_final = (
            _squeeze2d(
                self.mir_img
            )
        )

        self.if_final = (
            _squeeze2d(
                self.if_on_mir_image
            )
        )

        self.mir_disp = (
            _robust_normalize(
                self.mir_final,
                valid=np.isfinite(
                    self.mir_final
                ),
            )
        )

        self.if_disp = (
            _robust_normalize(
                self.if_final,
                valid=(
                    np.isfinite(
                        self.if_final
                    )
                    & (
                        self.if_final
                        != 0
                    )
                ),
            )
        )

    def _plot_final_mask(
        self,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        fig, ax = plt.subplots(
            figsize=(9, 12)
        )

        ax.imshow(
            self.mir_disp,
            cmap="gray",
        )

        if self.mask_final.any():
            ax.contour(
                self.mask_final,
                levels=[0.5],
                linewidths=1.0,
            )

        ax.set_title(
            "FINAL — refined plaque mask on MIR pixel grid"
        )

        ax.axis("off")

        self._finish_figure(
            fig,
            "final_plaque_mask_on_mir_grid",
        )

    def _plot_final_six_panel(
        self,
    ):
        if not (
            self.save_figures
            or self.show_figures
        ):
            return

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 12),
            constrained_layout=True,
        )

        axes[0, 0].imshow(
            self.mir_disp,
            cmap="gray",
        )
        axes[0, 0].set_title(
            "Aligned MIR — fixed final reference grid"
        )
        axes[0, 0].axis("off")

        axes[0, 1].imshow(
            self.if_disp,
            cmap="magma",
        )
        axes[0, 1].set_title(
            "IF on MIR grid"
        )
        axes[0, 1].axis("off")

        axes[0, 2].imshow(
            self.mask_final,
            cmap="gray",
        )
        axes[0, 2].set_title(
            "Plaque mask on MIR grid"
        )
        axes[0, 2].axis("off")

        axes[1, 0].imshow(
            self.mir_disp,
            cmap="gray",
        )
        axes[1, 0].imshow(
            self.if_disp,
            cmap="magma",
            alpha=0.38,
        )
        axes[1, 0].set_title(
            "Overlay: MIR + IF"
        )
        axes[1, 0].axis("off")

        axes[1, 1].imshow(
            self.mir_disp,
            cmap="gray",
        )
        axes[1, 1].contour(
            self.mask_final.astype(
                float
            ),
            levels=[0.5],
            colors="cyan",
            linewidths=0.8,
        )
        axes[1, 1].set_title(
            "Overlay: MIR + plaque-mask contour"
        )
        axes[1, 1].axis("off")

        axes[1, 2].imshow(
            self.mir_disp,
            cmap="gray",
        )
        axes[1, 2].imshow(
            self.if_disp,
            cmap="magma",
            alpha=0.30,
        )
        axes[1, 2].contour(
            self.mask_final.astype(
                float
            ),
            levels=[0.5],
            colors="lime",
            linewidths=0.8,
        )
        axes[1, 2].set_title(
            "Overlay: MIR + IF + plaque-mask contour"
        )
        axes[1, 2].axis("off")

        self._finish_figure(
            fig,
            "final_mir_if_mask_validation",
        )

    def _plot_plaque_zooms(
        self,
    ):
        """
        Show selected plaque-mask connected regions for final visual QC.

        Selection is controlled by ``plaque_zoom_selection``:

        ``largest``
            Historical behavior. Show the largest connected components.

        ``random``
            Show a random subset without replacement. ``plaque_zoom_seed``
            makes the subset exactly reproducible.

        ``mixed``
            Show approximately half of the largest plaques and fill the
            remaining rows with a seeded-random sample from the other plaques.

        Random sampling uses a local ``numpy.random.Generator``. Therefore the
        plaque QC seed does not modify NumPy's global random state and cannot
        influence the registration/refinement calculations.
        """
        max_regions = self.plaque_zoom_count

        if max_regions == 0:
            self.plaque_zoom_qc = {
                "selection": self.plaque_zoom_selection,
                "seed": self.plaque_zoom_seed,
                "requested_count": 0,
                "displayed_count": 0,
                "total_components": None,
                "selected_regions": [],
                "status": "disabled_by_plaque_zoom_count",
            }
            return

        if not (
            self.save_figures
            or self.show_figures
        ):
            self.plaque_zoom_qc = {
                "selection": self.plaque_zoom_selection,
                "seed": self.plaque_zoom_seed,
                "requested_count": int(max_regions),
                "displayed_count": 0,
                "total_components": None,
                "selected_regions": [],
                "status": "disabled_by_figure_output_settings",
            }
            return

        (
            plaque_labeled,
            n_plaque,
        ) = ndi.label(
            self.mask_final
        )

        if n_plaque == 0:
            self.plaque_zoom_qc = {
                "selection": self.plaque_zoom_selection,
                "seed": self.plaque_zoom_seed,
                "requested_count": int(max_regions),
                "displayed_count": 0,
                "total_components": 0,
                "selected_regions": [],
                "status": "no_plaque_components",
            }
            return

        objects = ndi.find_objects(
            plaque_labeled
        )

        regions = []

        for (
            label_id,
            region_slice,
        ) in enumerate(
            objects,
            start=1,
        ):
            if region_slice is None:
                continue

            region_mask = (
                plaque_labeled[
                    region_slice
                ]
                == label_id
            )

            area = int(
                region_mask.sum()
            )

            regions.append({
                "label": int(label_id),
                "slice": region_slice,
                "area": area,
            })

        # Keep a stable size ranking as the base ordering. This makes both
        # 'largest' and seeded random/mixed selection reproducible for the same
        # final mask. The component label is a deterministic tie-breaker.
        regions = sorted(
            regions,
            key=lambda item: (
                -item["area"],
                item["label"],
            ),
        )

        n_show = min(
            int(max_regions),
            len(regions),
        )

        rng = np.random.default_rng(
            self.plaque_zoom_seed
        )

        selected_regions = []

        if self.plaque_zoom_selection == "largest":
            for region in regions[:n_show]:
                selected = dict(region)
                selected["selection_source"] = "largest"
                selected_regions.append(selected)

        elif self.plaque_zoom_selection == "random":
            selected_indices = rng.choice(
                len(regions),
                size=n_show,
                replace=False,
            )

            for index in selected_indices:
                selected = dict(
                    regions[int(index)]
                )
                selected["selection_source"] = "random"
                selected_regions.append(selected)

        elif self.plaque_zoom_selection == "mixed":
            # With an even count this is exactly 50/50. With an odd count the
            # extra row is assigned to the largest-plaque group.
            n_largest = min(
                (n_show + 1) // 2,
                len(regions),
            )
            n_random = n_show - n_largest

            for region in regions[:n_largest]:
                selected = dict(region)
                selected["selection_source"] = "largest"
                selected_regions.append(selected)

            remaining = regions[n_largest:]

            if n_random > 0:
                random_indices = rng.choice(
                    len(remaining),
                    size=n_random,
                    replace=False,
                )

                for index in random_indices:
                    selected = dict(
                        remaining[int(index)]
                    )
                    selected["selection_source"] = "random"
                    selected_regions.append(selected)

        # Save compact, JSON-serializable provenance before plotting.
        self.plaque_zoom_qc = {
            "selection": self.plaque_zoom_selection,
            "seed": self.plaque_zoom_seed,
            "requested_count": int(max_regions),
            "displayed_count": int(len(selected_regions)),
            "total_components": int(len(regions)),
            "selected_regions": [
                {
                    "figure_row": int(row + 1),
                    "component_label": int(region["label"]),
                    "area_pixels": int(region["area"]),
                    "selection_source": str(
                        region["selection_source"]
                    ),
                }
                for row, region in enumerate(
                    selected_regions
                )
            ],
            "status": "generated",
        }

        mir_spacing_x_um = (
            float(
                self.mir_img.GetSpacing()[0]
            )
            * 1000.0
        )

        mir_spacing_y_um = (
            float(
                self.mir_img.GetSpacing()[1]
            )
            * 1000.0
        )

        half_width_px = max(
            25,
            int(
                np.ceil(
                    (
                        3.0
                        * self.plaque_radius_um
                    )
                    / mir_spacing_x_um
                )
            ),
        )

        half_height_px = max(
            25,
            int(
                np.ceil(
                    (
                        3.0
                        * self.plaque_radius_um
                    )
                    / mir_spacing_y_um
                )
            ),
        )

        fig, axes = plt.subplots(
            n_show,
            3,
            figsize=(
                15,
                4.3 * n_show,
            ),
            constrained_layout=True,
        )

        if n_show == 1:
            axes = np.asarray(
                [axes]
            )

        for (
            row,
            region,
        ) in enumerate(
            selected_regions
        ):
            region_slice = region[
                "slice"
            ]

            y_start = (
                region_slice[0].start
            )
            y_stop = (
                region_slice[0].stop
            )
            x_start = (
                region_slice[1].start
            )
            x_stop = (
                region_slice[1].stop
            )

            center_y = int(
                round(
                    (
                        y_start
                        + y_stop
                        - 1
                    )
                    / 2
                )
            )

            center_x = int(
                round(
                    (
                        x_start
                        + x_stop
                        - 1
                    )
                    / 2
                )
            )

            y0 = max(
                0,
                (
                    center_y
                    - half_height_px
                ),
            )

            y1 = min(
                self.mask_final.shape[0],
                (
                    center_y
                    + half_height_px
                ),
            )

            x0 = max(
                0,
                (
                    center_x
                    - half_width_px
                ),
            )

            x1 = min(
                self.mask_final.shape[1],
                (
                    center_x
                    + half_width_px
                ),
            )

            mir_crop = (
                self.mir_disp[
                    y0:y1,
                    x0:x1,
                ]
            )

            if_crop = (
                self.if_disp[
                    y0:y1,
                    x0:x1,
                ]
            )

            mask_crop = (
                self.mask_final[
                    y0:y1,
                    x0:x1,
                ]
            )

            area = region[
                "area"
            ]
            component_label = region[
                "label"
            ]
            selection_source = region[
                "selection_source"
            ]

            ax = axes[
                row,
                0,
            ]

            ax.imshow(
                mir_crop,
                cmap="gray",
            )

            ax.contour(
                mask_crop.astype(
                    float
                ),
                levels=[0.5],
                colors="cyan",
                linewidths=1.0,
            )

            ax.set_title(
                f"Plaque QC {row + 1}: MIR crop\n"
                f"component={component_label}, "
                f"area={area} px, "
                f"source={selection_source}"
            )

            ax.axis("off")

            ax = axes[
                row,
                1,
            ]

            ax.imshow(
                if_crop,
                cmap="magma",
            )

            ax.contour(
                mask_crop.astype(
                    float
                ),
                levels=[0.5],
                colors="cyan",
                linewidths=1.0,
            )

            ax.set_title(
                f"Plaque QC {row + 1}: IF crop"
            )

            ax.axis("off")

            ax = axes[
                row,
                2,
            ]

            ax.imshow(
                mir_crop,
                cmap="gray",
            )

            ax.imshow(
                if_crop,
                cmap="magma",
                alpha=0.35,
            )

            ax.contour(
                mask_crop.astype(
                    float
                ),
                levels=[0.5],
                colors="lime",
                linewidths=1.0,
            )

            ax.set_title(
                f"Plaque QC {row + 1}: "
                "MIR + IF + mask"
            )

            ax.axis("off")

        self._finish_figure(
            fig,
            "final_plaque_zoom_validation",
        )

    # =================================================================
    # Figure saving
    # =================================================================

    def _finish_figure(
        self,
        fig,
        name: str,
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
                name,
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

        plt.close(
            fig
        )

    # =================================================================
    # STEP 8 — summary, TXT report, manifest
    # =================================================================

    @staticmethod
    def _compact_qc(
        qc: Dict,
    ) -> Dict:
        return {
            "current_score": float(
                qc["zero"]["score"]
            ),
            "best_score": float(
                qc["fine_best"]["score"]
            ),
            "best_shift_x_um": float(
                qc["fine_best"][
                    "shift_x_um"
                ]
            ),
            "best_shift_y_um": float(
                qc["fine_best"][
                    "shift_y_um"
                ]
            ),
            "best_shift_magnitude_um": float(
                qc[
                    "correction_magnitude_um"
                ]
            ),
            "best_rotation_deg": float(
                qc["fine_best"][
                    "angle_deg"
                ]
            ),
            "score_gain_vs_current": float(
                qc[
                    "score_gain_vs_zero"
                ]
            ),
            "gradient_ncc_current": float(
                qc["zero"]["ncc"]
            ),
            "edge_dice_current": float(
                qc["zero"][
                    "edge_dice"
                ]
            ),
        }

    def _build_and_write_summary(
        self,
    ):
        if (
            self.if_mask_same_grid
            and self.mask_nonempty
            and self.initial_alignment_ok
            and self.direct_match_trustworthy
            and self.correction_accepted
            and self.if_pixels_unchanged
            and self.mask_pixels_unchanged
            and self.if_on_mir_grid
            and self.mask_on_mir_grid
        ):
            if self.correction_applied:
                self.pipeline_status = (
                    "READY_AFTER_AUTOMATIC_REFINEMENT"
                )
            else:
                self.pipeline_status = (
                    "READY_WITH_ORIGINAL_INITIAL ALIGNMENT_PLACEMENT"
                )
        else:
            self.pipeline_status = (
                "REJECTED"
            )

        self.summary = {
            "sample": self.sample_name,
            "pipeline_status": (
                self.pipeline_status
            ),
            "plaque_radius_um": (
                self.plaque_radius_um
            ),
            "preflight": {
                "if_mask_same_grid": bool(
                    self.if_mask_same_grid
                ),
                "mask_nonempty": bool(
                    self.mask_nonempty
                ),
            },
            "gross_alignment_gate": {
                "initial_alignment_ok": bool(
                    self.initial_alignment_ok
                ),
                "bf_mir_checks": (
                    self.bf_mir_gross_checks
                ),
                "bf_if_checks": (
                    self.bf_if_gross_checks
                ),
            },
            "direct_evidence_gate": {
                "structure_ok": bool(
                    self.direct_structure_ok
                ),
                "local_error_ok": bool(
                    self.direct_local_error_ok
                ),
                "match_trustworthy": bool(
                    self.direct_match_trustworthy
                ),
                "correction_recommended": bool(
                    self.correction_recommended
                ),
            },
            "post_correction_gate": {
                "accepted": bool(
                    self.correction_accepted
                ),
                "max_residual_shift_um": (
                    self.post_max_residual_shift_um
                ),
                "max_rotation_deg": (
                    self.post_max_rot_deg
                ),
                "max_remaining_score_gain": (
                    self.post_max_score_gain
                ),
            },
            "bf_mir": {
                "status": (
                    self.bf_mir_status
                ),
                **self._compact_qc(
                    self.bf_mir_qc
                ),
            },
            "bf_if": {
                "status": (
                    self.bf_if_status
                ),
                **self._compact_qc(
                    self.bf_if_qc
                ),
            },
            "direct_mir_if_before": (
                self._compact_qc(
                    self.mir_if_direct_qc
                )
            ),
            "direct_translation_only_before": (
                self._compact_qc(
                    self.mir_if_translation_only_qc
                )
            ),
            "rotation_necessity": {
                "rigid_best_score": float(
                    self.mir_if_direct_qc[
                        "fine_best"
                    ]["score"]
                ),
                "translation_only_best_score": float(
                    self.mir_if_translation_only_qc[
                        "fine_best"
                    ]["score"]
                ),
                "extra_gain_from_rotation": float(
                    self.rotation_extra_gain
                ),
                "min_extra_gain_required": float(
                    self.rotation_min_extra_gain
                ),
                "rotation_required": bool(
                    self.rotation_required
                ),
                "selected_refinement_mode": (
                    self.selected_refinement_mode
                ),
                "rigid_candidate_dx_um": float(
                    self.rigid_candidate_dx_um
                ),
                "rigid_candidate_dy_um": float(
                    self.rigid_candidate_dy_um
                ),
                "rigid_candidate_rotation_deg": float(
                    self.rigid_candidate_rot_deg
                ),
                "translation_candidate_dx_um": float(
                    self.translation_candidate_dx_um
                ),
                "translation_candidate_dy_um": float(
                    self.translation_candidate_dy_um
                ),
            },
            "direct_search_boundary_flags": (
                self.direct_boundary
            ),
            "rigid_refinement_safe": bool(
                self.rigid_refinement_safe
            ),
            "automatic_correction_enabled": bool(
                self.auto_refine
            ),
            "correction_applied": bool(
                self.correction_applied
            ),
            "selected_refinement_mode": (
                self.selected_refinement_mode
            ),
            "rotation_required": bool(
                self.rotation_required
            ),
            "applied_dx_um": (
                self.direct_dx_um
                if self.correction_applied
                else 0.0
            ),
            "applied_dy_um": (
                self.direct_dy_um
                if self.correction_applied
                else 0.0
            ),
            "applied_rotation_deg": (
                self.direct_rot_deg
                if self.correction_applied
                else 0.0
            ),
            "direct_mir_if_after": (
                self._compact_qc(
                    self.mir_if_post_qc
                )
            ),
            "if_pixels_unchanged": bool(
                self.if_pixels_unchanged
            ),
            "mask_pixels_unchanged": bool(
                self.mask_pixels_unchanged
            ),
            "if_on_mir_grid": bool(
                self.if_on_mir_grid
            ),
            "mask_on_mir_grid": bool(
                self.mask_on_mir_grid
            ),
            "plaque_positive_mir_pixels": int(
                self.plaque_positive_mir_pixels
            ),
            "plaque_zoom_qc": self.plaque_zoom_qc,
            "outputs": {
                "if_refined": (
                    str(
                        self.if_refined_path
                    )
                    if self.correction_applied
                    else None
                ),
                "mask_refined": (
                    str(
                        self.mask_refined_path
                    )
                    if self.correction_applied
                    else None
                ),
                "if_on_mir_grid": str(
                    self.if_on_mir_path
                ),
                "mask_on_mir_grid": str(
                    self.mask_on_mir_path
                ),
                "figures_dir": str(
                    self.figures_dir
                ),
            },
        }

        self.summary["parameters"] = self.GetParameters()
        self.summary["stage"] = "complete"
        self.summary["failure_reason"] = None
        self.summary["outputs"]["summary_json"] = str(
            self.summary_json_path
        )
        self.summary["outputs"]["figures_dir"] = (
            str(self.figures_dir)
            if self.save_figures
            else None
        )

        self.summary_json_path.write_text(
            json.dumps(
                self.summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _qc_lines(
        self,
        name: str,
        qc: Dict,
    ):
        best = qc[
            "fine_best"
        ]
        zero = qc[
            "zero"
        ]

        return [
            name,
            (
                "  current score          : "
                f"{zero['score']}"
            ),
            (
                "  current gradient NCC   : "
                f"{zero['ncc']}"
            ),
            (
                "  current edge Dice      : "
                f"{zero['edge_dice']}"
            ),
            (
                "  best score             : "
                f"{best['score']}"
            ),
            (
                "  best shift X [um]      : "
                f"{best['shift_x_um']}"
            ),
            (
                "  best shift Y [um]      : "
                f"{best['shift_y_um']}"
            ),
            (
                "  best shift magnitude   : "
                f"{qc['correction_magnitude_um']}"
            ),
            (
                "  best rotation [deg]    : "
                f"{best['angle_deg']}"
            ),
            (
                "  score gain             : "
                f"{qc['score_gain_vs_zero']}"
            ),
            "",
        ]

    def _write_json_checkpoint(
        self,
        stage: str,
        run_status: str = "RUNNING",
        note: Optional[str] = None,
    ):
        """
        Write a partial JSON status checkpoint.

        This is intentionally the only summary/report format generated by the
        class. It is also called before fail-fast exceptions, so a rejected run
        still leaves a machine-readable explanation on disk.
        """
        payload = {
            "sample": self.sample_name,
            "pipeline_status": run_status,
            "stage": stage,
            "failure_reason": note,
            "parameters": self.GetParameters(),
            "outputs": {
                "summary_json": str(
                    self.summary_json_path
                ),
                "figures_dir": (
                    str(self.figures_dir)
                    if self.save_figures
                    else None
                ),
            },
        }

        if hasattr(
            self,
            "if_mask_same_grid",
        ):
            payload["preflight"] = {
                "if_mask_same_grid": bool(
                    self.if_mask_same_grid
                ),
                "mask_nonempty": bool(
                    self.mask_nonempty
                ),
            }

        if hasattr(
            self,
            "bf_mir_qc",
        ):
            payload["bf_mir"] = {
                "status": getattr(
                    self,
                    "bf_mir_status",
                    None,
                ),
                **self._compact_qc(
                    self.bf_mir_qc
                ),
            }

        if hasattr(
            self,
            "bf_if_qc",
        ):
            payload["bf_if"] = {
                "status": getattr(
                    self,
                    "bf_if_status",
                    None,
                ),
                **self._compact_qc(
                    self.bf_if_qc
                ),
            }

        if hasattr(
            self,
            "initial_alignment_ok",
        ):
            payload[
                "gross_alignment_gate"
            ] = {
                "initial_alignment_ok": bool(
                    self.initial_alignment_ok
                ),
                "bf_mir_checks": (
                    self.bf_mir_gross_checks
                ),
                "bf_if_checks": (
                    self.bf_if_gross_checks
                ),
            }

        if hasattr(
            self,
            "mir_if_direct_qc",
        ):
            payload[
                "direct_mir_if_before"
            ] = self._compact_qc(
                self.mir_if_direct_qc
            )

        if hasattr(
            self,
            "mir_if_translation_only_qc",
        ):
            payload[
                "direct_translation_only_before"
            ] = self._compact_qc(
                self.mir_if_translation_only_qc
            )

        if hasattr(
            self,
            "rotation_required",
        ):
            payload[
                "rotation_necessity"
            ] = {
                "rigid_best_score": float(
                    self.mir_if_direct_qc[
                        "fine_best"
                    ]["score"]
                ),
                "translation_only_best_score": float(
                    self.mir_if_translation_only_qc[
                        "fine_best"
                    ]["score"]
                ),
                "extra_gain_from_rotation": float(
                    self.rotation_extra_gain
                ),
                "min_extra_gain_required": float(
                    self.rotation_min_extra_gain
                ),
                "rotation_required": bool(
                    self.rotation_required
                ),
                "selected_refinement_mode": (
                    self.selected_refinement_mode
                ),
            }

        if hasattr(
            self,
            "direct_match_trustworthy",
        ):
            payload[
                "direct_evidence_gate"
            ] = {
                "structure_ok": bool(
                    self.direct_structure_ok
                ),
                "local_error_ok": bool(
                    self.direct_local_error_ok
                ),
                "match_trustworthy": bool(
                    self.direct_match_trustworthy
                ),
                "correction_recommended": bool(
                    self.correction_recommended
                ),
                "search_boundary_flags": (
                    self.direct_boundary
                ),
            }

        if hasattr(
            self,
            "correction_applied",
        ):
            payload["correction"] = {
                "applied": bool(
                    self.correction_applied
                ),
                "selected_refinement_mode": (
                    self.selected_refinement_mode
                ),
                "rotation_required": bool(
                    self.rotation_required
                ),
                "dx_um": float(
                    self.direct_dx_um
                    if self.correction_applied
                    else 0.0
                ),
                "dy_um": float(
                    self.direct_dy_um
                    if self.correction_applied
                    else 0.0
                ),
                "rotation_deg": float(
                    self.direct_rot_deg
                    if self.correction_applied
                    else 0.0
                ),
            }

        if hasattr(
            self,
            "mir_if_post_qc",
        ):
            payload[
                "direct_mir_if_after"
            ] = self._compact_qc(
                self.mir_if_post_qc
            )

        if hasattr(
            self,
            "correction_accepted",
        ):
            payload[
                "post_correction_gate"
            ] = {
                "accepted": bool(
                    self.correction_accepted
                ),
                "score_not_worse": bool(
                    self.score_not_worse
                ),
                "remaining_shift_um": float(
                    self.post_best_shift_um
                ),
                "remaining_rotation_deg": float(
                    self.post_best_rot_deg
                ),
                "remaining_score_gain": float(
                    self.post_gain
                ),
            }

        if hasattr(
            self,
            "if_pixels_unchanged",
        ):
            payload[
                "pixel_and_grid_checks"
            ] = {
                "if_pixels_unchanged": bool(
                    self.if_pixels_unchanged
                ),
                "mask_pixels_unchanged": bool(
                    self.mask_pixels_unchanged
                ),
                "if_on_mir_grid": (
                    bool(self.if_on_mir_grid)
                    if hasattr(
                        self,
                        "if_on_mir_grid",
                    )
                    else None
                ),
                "mask_on_mir_grid": (
                    bool(self.mask_on_mir_grid)
                    if hasattr(
                        self,
                        "mask_on_mir_grid",
                    )
                    else None
                ),
            }

        if hasattr(
            self,
            "plaque_positive_mir_pixels",
        ):
            payload[
                "plaque_positive_mir_pixels"
            ] = int(
                self.plaque_positive_mir_pixels
            )

        if (
            hasattr(self, "plaque_zoom_qc")
            and self.plaque_zoom_qc is not None
        ):
            payload["plaque_zoom_qc"] = (
                self.plaque_zoom_qc
            )

        self.summary_json_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )
