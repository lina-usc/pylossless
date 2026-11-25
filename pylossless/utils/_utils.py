# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Utility Functions for running the Lossless Pipeline."""

import numpy as np
import pandas as pd
from mne.utils import check_version, logger

from .html import _sum_flagged_times


# MNE renamed the ``standard_*`` montages to ``colin27_*`` in MNE 1.13.0
# When the dependency declaration is raised to ``mne>=1.13``
# this triage can be removed.
_RENAMED_STANDARD_MONTAGES = {
    "standard_1005": "colin27_1005",
    "standard_1020": "colin27_1020",
    "standard_alphabetic": "colin27_alphabetic",
    "standard_postfixed": "colin27_postfixed",
    "standard_prefixed": "colin27_prefixed",
    "standard_primed": "colin27_primed",
}


def _resolve_montage_name(montage_name):
    """Return the montage name valid for the installed MNE version.

    Parameters
    ----------
    montage_name : str | object
        A builtin montage name, e.g. ``"standard_1020"``, or any other
        object (a ``DigMontage``, ``None``, ...), which is returned
        unchanged.

    Returns
    -------
    str | object
        ``"colin27_..."`` instead of ``"standard_..."`` on MNE >= 1.13,
        where the ``standard_*`` names were renamed; the input unchanged
        on MNE < 1.13, where the ``colin27_*`` names do not exist.
    """
    if isinstance(montage_name, str) and check_version("mne", "1.13"):
        return _RENAMED_STANDARD_MONTAGES.get(montage_name, montage_name)
    return montage_name


def _icalabel_to_data_frame(ica):
    """Export IClabels to pandas DataFrame."""
    ic_type = [""] * ica.n_components_
    for label, comps in ica.labels_.items():
        for comp in comps:
            ic_type[comp] = label

    # Create TSV.
    return pd.DataFrame(
        dict(
            component=ica._ica_names,
            annotator=["ic_label"] * ica.n_components_,
            ic_type=ic_type,
            confidence=ica.labels_scores_.max(1),
        )
    )


def _report_flagged_epochs(raw, flag):
    times = _sum_flagged_times(raw, flag)[flag]
    if not times:
        times = 0
    else:
        logger.info(f"📋 LOSSLESS: {np.round(times, 2)} second(s) flagged as {flag}")
