# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Utility Functions for running the Lossless Pipeline."""

import pandas as pd


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
