import numpy as np


def _setup_vmin_vmax(data, vmin, vmax, norm=False):
    """Handle vmin and vmax parameters for visualizing topomaps.

    This is a simplified copy of mne.viz.utils._setup_vmin_vmax.
    https://github.com/mne-tools/mne-python/blob/main/mne/viz/utils.py

    Notes
    -----
    For the normal use-case (when `vmin` and `vmax` are None), the parameter
    `norm` drives the computation. When norm=False, data is supposed to come
    from a mag and the output tuple (vmin, vmax) is symmetric range
    (-x, x) where x is the max(abs(data)). When norm=True (a.k.a. data is the
    L2 norm of a gradiometer pair) the output tuple corresponds to (0, x).

    in the MNE version vmin and vmax can be callables that drive the operation,
    but for the sake of simplicity this was not copied over.
    """
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        vmin = 0.0 if norm else -vmax
    return vmin, vmax
