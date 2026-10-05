def test_import_mir_if_alignment_qc():
    import pymirimaging as mi

    assert hasattr(
        mi,
        "MIRIFAlignmentQC",
    )


def test_mir_if_alignment_qc_parameter_help():
    import pymirimaging as mi

    help_text = (
        mi.MIRIFAlignmentQC
        .GetParameterHelp()
    )

    assert (
        "plaque_zoom_count"
        in help_text
    )

    assert (
        "min_gradient_ncc"
        in help_text
    )

    assert (
        "min_edge_dice"
        in help_text
    )


def test_mir_if_alignment_qc_defaults_without_loading_images(
    tmp_path,
):
    import pymirimaging as mi

    qc = mi.MIRIFAlignmentQC(
        brightfield="bf.nrrd",
        mir_aligned="mir.nrrd",
        if_aligned="if.nrrd",
        mask_aligned="mask.nrrd",
        output_dir=tmp_path,
        sample_name="test_sample",
        save_figures=False,
        show_figures=False,
    )

    params = qc.GetParameters()

    assert (
        params["plaque_radius_um"]
        == 40.0
    )

    assert (
        params["plaque_zoom_count"]
        == 6
    )

    assert (
        params["auto_refine"]
        is True
    )

    assert (
        params["min_gradient_ncc"]
        == 0.10
    )

    assert (
        params["min_edge_dice"]
        == 0.40
    )

    assert (
        params["direct_max_shift_um"]
        == 40.0
    )

    assert (
        params["post_max_residual_shift_um"]
        == 10.0
    )
