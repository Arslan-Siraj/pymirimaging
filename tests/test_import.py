def test_import_package():
    import pymirimaging as mi

    assert hasattr(mi, "ZarrSpectrumReader")
    assert hasattr(mi, "NRRDSpectrumReader")
    assert hasattr(mi, "ImageReader")
