import importlib


# A mapping from import name to package name (on PyPI) when the package name
# is different.
_INSTALL_MAPPING = {
    "codespell_lib": "codespell",
    "openneuro": "openneuro-py",
    "pytest_cov": "pytest-cov",
    "sklearn": "scikit-learn",
}


def import_optional_dependency(
    name,
    extra=None,
    raise_error=True,
):
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str | None
        Additional text to include in the ImportError message. Default is None, which
        means no additional text.
    raise_error : bool
        What to do when a dependency is not found.
        * True : Raise an ImportError.
        * False: Return None.

    Returns
    -------
    module : Module | None
        The imported module when found.
        None is returned when the package is not found and raise_error is False.
    """
    package_name = _INSTALL_MAPPING.get(name, name)
    if importlib.util.find_spec(name) is None:
        if raise_error:
            raise ImportError(
                f"Missing optional dependency '{package_name}'. {extra} Use pip or "
                f"conda to install {package_name}."
            )
        else:
            return None
    return importlib.import_module(name)
