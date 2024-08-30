import re
from pathlib import Path

import pytest

@pytest.mark.skip(reason="This test is not yet implemented.")
def test_toplevel_imports():
    """Test that there are no optional dependencies at the top level."""
    optional_deps = []
    with (Path(".").parent.parent.parent / "requirements.txt") as file:
        for line in file.read_text().split("\n"):
            if line and not line.startswith("#"):
                # regex for any one or two of the characters <, >, =, !
                regex = r"[<>=!]{1,2}"
                optional_deps.append(re.split(regex, line)[0])
    pass

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
