# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Utility Functions for running the Lossless Pipeline."""

from mne_icalabel.config import ICLABEL_LABELS_TO_MNE
import pandas as pd


def _icalabel_to_data_frame(ica):
    """Export IClabels to pandas DataFrame."""
    # initialize status, description and IC type
    status = ["good"] * ica.n_components_
    status_description = ["n/a"] * ica.n_components_
    ic_type = ["n/a"] * ica.n_components_

    # extract the component labels if they are present in the ICA instance
    if ica.labels_:
        for label, comps in ica.labels_.items():
            this_status = "good" if label == "brain" else "bad"
            if label in ICLABEL_LABELS_TO_MNE.values():
                for comp in comps:
                    status[comp] = this_status
                    ic_type[comp] = label

    # Create TSV.
    return pd.DataFrame(
        dict(
            component=list(range(ica.n_components_)),
            type=["ica"] * ica.n_components_,
            description=["Independent Component"] * ica.n_components_,
            status=status,
            status_description=status_description,
            annotate_method=["n/a"] * ica.n_components_,
            annotate_author=["n/a"] * ica.n_components_,
            ic_type=ic_type,
        )
    )
