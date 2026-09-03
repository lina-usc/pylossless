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


def test_validate_kwargs():
    """Test checking keyword arguments against a function signature."""
    from pylossless.utils import check

    def func(required, optional=None, *, keyword_only=None):
        return required, optional, keyword_only

    kwargs = {"optional": 1, "keyword_only": 2}
    assert check._validate_kwargs(func, kwargs, name="func") == kwargs

    with pytest.raises(TypeError, match="Invalid parameter\\(s\\) for func: extra"):
        check._validate_kwargs(func, {"optional": 1, "extra": 2}, name="func")

    with pytest.warns(
        RuntimeWarning,
        match="Ignoring parameter\\(s\\) unsupported by func: extra",
    ):
        filtered_kwargs = check._validate_kwargs(
            func,
            {"required": 0, "optional": 1, "extra": 2},
            name="func",
            exclude=("required",),
            strict=False,
        )
    assert filtered_kwargs == {"optional": 1}

    def func_with_kwargs(**kwargs):
        return kwargs

    passthrough_kwargs = {"unknown": 1}
    assert (
        check._validate_kwargs(func_with_kwargs, passthrough_kwargs, name="func")
        is passthrough_kwargs
    )
