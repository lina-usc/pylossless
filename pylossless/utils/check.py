import importlib
import inspect
import warnings


# A mapping from import name to package name (on PyPI) when the package name
# is different.
_INSTALL_MAPPING = {
    "amica": "amica-python",
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


def _validate_kwargs(func, kwargs, *, name, exclude=(), strict=True):
    """Validate kwargs for func, optionally dropping invalid parameters."""
    signature = inspect.signature(func)
    # If a function accepts **kwargs, any keyword arguments are valid
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return kwargs

    valid_params = {
        param_name
        for param_name, param in signature.parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
    }
    valid_params.difference_update(exclude)

    invalid_params = sorted(set(kwargs) - valid_params)
    if invalid_params:
        invalid_str = ", ".join(invalid_params)
        valid_str = ", ".join(sorted(valid_params))
        if not strict:
            warnings.warn(
                f"Ignoring parameter(s) unsupported by {name}: {invalid_str}. "
                f"Valid parameters are: {valid_str}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                param_name: param_value
                for param_name, param_value in kwargs.items()
                if param_name in valid_params
            }
        raise TypeError(
            f"Invalid parameter(s) for {name}: {invalid_str}. "
            f"Valid parameters are: {valid_str}."
        )
    return kwargs
