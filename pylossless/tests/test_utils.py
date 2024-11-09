import pytest

def test_import_optional_dependency():
    """Test the import_optional_dependency function."""
    from pylossless.utils import check

    # Test the case where the package is not installed.
    package = "sdgssfersfsdesdfsefsdfsdt"
    with pytest.raises(ImportError, match=f"Missing optional dependency '{package}'."):
        # Choosing a package that will probably never be added to the requirements!
        # We also choose a name of a package that is likely not to exist at all
        # to avoid the corresponding package is installed in the development
        # environment of developers.
        check.import_optional_dependency(package)

    # Test the case where the package is installed.
    mne = check.import_optional_dependency("mne", raise_error=False)
    assert mne is not None

    # Test the where case package is not installed but we don't want to raise an error.
    ret_val = check.import_optional_dependency(package, raise_error=False)
    assert ret_val is None
