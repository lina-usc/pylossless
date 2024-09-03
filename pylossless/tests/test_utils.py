import pytest

def test_import_optional_dependency():
    """Test the import_optional_dependency function."""
    from pylossless.utils import check

    # Test the case where the package is not installed.
    with pytest.raises(ImportError, match="Missing optional dependency 'astropy'."):
        # choosing a package that will probably never be added to the requirements!
        check.import_optional_dependency("astropy")

    # Test the case where the package is installed.
    mne = check.import_optional_dependency("mne", raise_error=False)
    assert mne is not None

    # Test the where case package is not installed but we don't want to raise an error.
    astropy = check.import_optional_dependency("astropy", raise_error=False)
    assert astropy is None
