# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Auxiliary functions for the Lossless Pipeline."""

from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
import scipy
from scipy.spatial import distance_matrix
from tqdm import tqdm

import mne
from mne.coreg import Coregistration
from mne.utils import warn


def epochs_to_xr(epochs, kind="ch", ica=None):
    """Create an Xarray DataArray from an instance of mne.Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        an instance of mne.Epochs
    kind : string
        The name to be passed into the `coords` argument of xr.DataArray
        corresponding to the channel dimension of the epochs object.
        Must be ``'ch'`` or ``'ic'``.
    ica : mne.preprocessing.ICA
        If not ``None``, should be an instance of mne.preprocessing.ICA
        from which to pull the names of the ICA components.

    Returns
    -------
    xarray.DataArray
        an instance of xarray.DataArray, with dimensions ``'epochs'``,
        ``'time'`` (samples), and either ``'ch'`` (channels) or ``'ic'``
        (independent components).
    """
    if kind == "ch":
        data = epochs.get_data()  # n_epochs, n_channels, n_times
        names = epochs.ch_names
    elif kind == "ic":
        data = ica.get_sources(epochs).get_data()
        names = ica._ica_names

    else:
        raise ValueError("The argument kind must be equal to 'ch' or 'ic'.")

    return xr.DataArray(
        data,
        coords={"epoch": np.arange(data.shape[0]), kind: names, "time": epochs.times},
    )


def get_operate_dim(array, flag_dim):
    """Get the xarray.DataArray dimension to flag for a pipeline method.

    Parameters
    ----------
    array : xarray.DataArray
        An instance of Xarray.DataArray that was constructed from an
        ``mne.Epochs`` object, using ``pylossless.pipeline.epochs_to_xr``.
        The ``array`` must be 2D.
    flag_dim : str
        Name of the dimension to remove in ``xarray.DataArray.dims``.
        Must be one of ``'epoch'``, ``'ch'``, or ``'ic'``.

    Returns
    -------
    list : list
        a list of the dimensions of the xarray.DataArray,
        excluding the dimension that the pipeline will conduct
        flagging operations on.
    """
    dims = list(array.dims)
    assert len(dims) == 2
    dims.remove(flag_dim)
    return dims[0]


def _get_outliers_quantile(array, dim, lower=0.25, upper=0.75, mid=0.5, k=3):
    """Calculate outliers for Epochs or Channels based on the IQR.

    Parameters
    ----------
    array : xr.DataArray
        Array of shape n_channels, n_epochs, representing the stdev across
        time (samples in epoch) for each channel/epoch pair.
    dim : str
        One of 'ch' or 'epoch'. The dimension to operate across.
    lower : float (default 0.75)
        The lower bound of the IQR
    upper : float (default 0.75)
        The upper bound of the IQR
    mid : float (default 0.5)
        The mid-point of the IQR
    k : int | float
        factor to multiply the IQR by.

    Returns
    -------
    Lower value threshold : xr.DataArray
        Vector of values (of size n_channels or n_epochs) to be considered
        as the lower threshold for outliers.
    Upper value threshold : xr.DataArray
        Vector of values (of size n_channels or n_epochs) to be considered the
        upper thresholds for outliers.
    """
    lower_val, mid_val, upper_val = array.quantile([lower, mid, upper], dim=dim)

    # Code below deviates from Tukeys method (Q2 +/- k(Q3-Q1))
    # because we need to account for distribution skewness.
    lower_dist = mid_val - lower_val
    upper_dist = upper_val - mid_val
    return mid_val - lower_dist * k, mid_val + upper_dist * k


def _get_outliers_trimmed(array, dim, trim=0.2, k=3):
    """Calculate outliers for Epochs or Channels based on the trimmed mean."""
    trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
    trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))
    m_dist = array.reduce(trim_mean, dim=dim)
    s_dist = array.reduce(trim_std, dim=dim)
    return m_dist - s_dist * k, m_dist + s_dist * k


def _detect_outliers(
    array,
    flag_dim="epoch",
    outlier_method="quantile",
    flag_crit=0.2,
    init_dir="both",
    outliers_kwargs=None,
):
    """Mark epochs, channels, or ICs as flagged for artefact.

    Parameters
    ----------
    array : xr.DataArray
        Array of shape n_channels, n_epochs, representing the stdev across
        time (samples in epoch) for each channel/epoch pair.
    dim : str
        One of 'ch' or 'epoch'. The dimension to operate across. For example
        if 'epoch', then detect epochs that are outliers.
    outlier_method : str (default quantile)
        one of 'quantile', 'trimmed', or 'fixed'.
    flag_crit : float
        Threshold (percentage) to consider an epoch or channel as bad. If
        operating across channels using default value, then if more then if
        the channel is an outlier in more than 20% of epochs, it will be
        flagged. if operating across epochs, then if more than 20% of channels
        are outliers in an epoch, it will be flagged as bad.
    init_dir : str
        One of 'pos', 'neg', or 'both'. Direction to test for outliers. If
        'pos', only detect outliers at the upper end of the distribution. If
        'neg', only detect outliers at the lower end of the distribution.
    outliers_kwargs : dict
        Set in the pipeline config. 'k', 'lower', and 'upper' kwargs can be
        passed to _get_outliers_quantile. 'k' can also be passed to
        _get_outliers_trimmed.

    Returns
    -------
    boolean xr.DataArray of shape n_epochs, n_times, where an epoch x channel
    coordinate is 1 if it is to be flagged as bad.

    """
    if outliers_kwargs is None:
        outliers_kwargs = {}

    # Computing lower and upper bounds for outlier detection
    operate_dim = get_operate_dim(array, flag_dim)

    if outlier_method == "quantile":
        l_out, u_out = _get_outliers_quantile(array, flag_dim, **outliers_kwargs)

    elif outlier_method == "trimmed":
        l_out, u_out = _get_outliers_trimmed(array, flag_dim, **outliers_kwargs)

    elif outlier_method == "fixed":
        l_out, u_out = outliers_kwargs["lower"], outliers_kwargs["upper"]

    else:
        raise ValueError(
            "outlier_method must be 'quantile', 'trimmed'"
            f", or 'fixed'. Got {outlier_method}"
        )

    # Calculating the proportion of outliers along dimension operate_dim
    # and marking items along dimension flag_dim if this number is
    # larger than
    outlier_mask = xr.zeros_like(array, dtype=bool)

    if init_dir == "pos" or init_dir == "both":  # for positive outliers
        outlier_mask = outlier_mask | (array > u_out)

    if init_dir == "neg" or init_dir == "both":  # for negative outliers
        outlier_mask = outlier_mask | (array < l_out)

    # average column of outlier_mask
    # drop quantile coord because it is no longer needed
    prop_outliers = outlier_mask.astype(float).mean(operate_dim)
    if "quantile" in list(prop_outliers.coords.keys()):
        prop_outliers = prop_outliers.drop_vars("quantile")
    return prop_outliers[prop_outliers > flag_crit].coords.to_index().values


def find_bads_by_threshold(epochs, threshold=5e-5):
    """Return channels with a standard deviation consistently above a fixed threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        an instance of mne.Epochs with a single channel type.
    threshold : float
        the threshold in volts. If the standard deviation of a channel's voltage
        variance at a specific epoch is above the threshold, then that channel x epoch
        will be flagged as an "outlier". If more than 20% of epochs are flagged as
        outliers for a specific channel, then that channel will be flagged as bad.
        Default threshold is 5e-5 (0.00005), i.e. 50 microvolts.

    Returns
    -------
    list
        a list of channel names that are considered outliers.

    Notes
    -----
    If you are having trouble converting between exponential notation and
    decimal notation, you can use the following code to convert between the two:

    >>> import numpy as np
    >>> threshold = 5e-5
    >>> with np.printoptions(suppress=True):
    ...     print(threshold)
    0.00005

    .. seealso::

        :func:`~pylossless.LosslessPipeline.flag_channels_fixed_threshold` to use
        this function within the lossless pipeline.

    Examples
    --------
    >>> import mne
    >>> import pylossless as ll
    >>> fname = mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif"
    >>> raw = mne.io.read_raw(fname, preload=True).pick("eeg")
    >>> raw.apply_function(lambda x: x * 3, picks=["EEG 001"]) # Make a noisy channel
    >>> epochs = mne.make_fixed_length_epochs(raw, preload=True)
    >>> bad_chs = ll.pipeline.find_bads_by_threshold(epochs)
    """
    # TODO: We should make this function handle multiple channel types.
    # TODO: but I'd like to avoid making a copy of the epochs object
    ch_types = np.unique(epochs.get_channel_types()).tolist()
    if len(ch_types) > 1:
        warn(
            f"The epochs object contains multiple channel types: {ch_types}.\n"
            " This will likely bias the results of the threshold detection."
            " Use the `mne.Epochs.pick` to select a single channel type."
        )
    bads = _threshold_volt_std(epochs, flag_dim="ch", threshold=threshold)
    return bads


def _threshold_volt_std(epochs, flag_dim, threshold=5e-5):
    """Detect epochs or channels whose voltage std is above threshold.

    Parameters
    ----------
    flag_dim : str
        The dimension to flag outlier in. 'ch' for channels, 'epoch'
        for epochs.
    threshold : float | tuple | list
        The threshold in volts. If the standard deviation of a channel's
        voltage variance at a specific epoch is above the threshold, then
        that channel x epoch will be flagged as an "outlier". If threshold
        is a single int or float, then it is treated as the upper threshold
            and the lower threshold is set to 0. Default is 5e-5, i.e.
            50 microvolts.
    """
    if isinstance(threshold, (tuple, list)):
        assert len(threshold) == 2
        l_out, u_out = threshold
        init_dir = "both"
    elif isinstance(threshold, (float, int)):
        l_out, u_out = (0, threshold)
        init_dir = "pos"
    else:
        raise ValueError(
            "threshold must be an int, float, or a list/tuple"
            f" of 2 int or float values. got {threshold}"
        )

    epochs_xr = epochs_to_xr(epochs, kind="ch")
    data_sd = epochs_xr.std("time")
    # Flag channels or epochs if their std is above
    # a fixed threshold.
    outliers_kwargs = dict(lower=l_out, upper=u_out)
    volt_outlier_inds = _detect_outliers(
        data_sd,
        flag_dim=flag_dim,
        outlier_method="fixed",
        init_dir=init_dir,
        outliers_kwargs=outliers_kwargs,
    )
    return volt_outlier_inds


def chan_neighbour_r(epochs, nneigbr, method):
    """Compute nearest Neighbor R.

    Parameters
    ----------
    epochs : mne.Epochs

    nneigbr : int
        Number of neighbours to compare in open interval

    method : str
        One of 'max', 'mean', or 'trimmean'. This is the function
        which aggregates the neighbours into one value.

    Returns
    -------
    Xarray : Xarray.DataArray
        An instance of Xarray.DataArray
    """
    chan_locs = pd.DataFrame(epochs.get_montage().get_positions()["ch_pos"]).T
    chan_dist = pd.DataFrame(
        distance_matrix(chan_locs, chan_locs),
        columns=chan_locs.index,
        index=chan_locs.index,
    )
    rank = chan_dist.rank("columns", ascending=True) - 1
    rank[rank == 0] = np.nan
    nearest_neighbor = pd.DataFrame(
        {
            ch_name: row.dropna().sort_values()[:nneigbr].index.values
            for ch_name, row in rank.iterrows()
        }
    ).T

    r_list = []
    for name, row in tqdm(list(nearest_neighbor.iterrows())):
        this_ch = epochs.get_data(name)
        nearest_chs = epochs.get_data(list(row.values))
        this_ch_xr = xr.DataArray(
            [this_ch * np.ones_like(nearest_chs)],
            dims=["ref_chan", "epoch", "channel", "time"],
            coords={
                "ref_chan": [name],
                "epoch": np.arange(len(epochs)),
                "channel": row.values.tolist(),
                "time": epochs.times,
            },
        )
        nearest_chs_xr = xr.DataArray(
            [nearest_chs],
            dims=["ref_chan", "epoch", "channel", "time"],
            coords={
                "ref_chan": [name],
                "epoch": np.arange(len(epochs)),
                "channel": row.values.tolist(),
                "time": epochs.times,
            },
        )
        r_list.append(xr.corr(this_ch_xr, nearest_chs_xr, dim=["time"]))

    c_neigbr_r = xr.concat(r_list, dim="ref_chan")

    if method == "max":
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).max(dim="channel")

    elif method == "mean":
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).mean(dim="channel")

    elif method == "trimmean":
        trim_mean_10 = partial(scipy.stats.trim_mean, proportiontocut=0.1)
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).reduce(
            trim_mean_10, dim="channel"
        )

    return m_neigbr_r.rename(ref_chan="ch")


def coregister(
    raw_edf,
    fiducials="estimated",  # get fiducials from fsaverage
    show_coreg=False,
    verbose=False,
):
    """Coregister Raw object to `'fsaverage'`.

    Parameters
    ----------
    raw_edf : mne.io.Raw
        an instance of `mne.io.Raw` to coregister.
    fiducials : str (default 'estimated')
        fiducials to use for coregistration. if `'estimated'`, gets fiducials
        from fsaverage.
    show_coreg : bool (default False)
        If True, shows the coregistration result in a plot.
    verbose : bool | str (default False)
        sets the logging level for `mne.Coregistration`.

    Returns
    -------
    coregistration | numpy.array
        a numpy array containing the coregistration trans values.
    """
    plot_kwargs = dict(
        subject="fsaverage", surfaces="head-dense", dig=True, show_axes=True
    )

    coreg = Coregistration(raw_edf.info, "fsaverage", fiducials=fiducials)
    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=verbose)

    if show_coreg:
        mne.viz.plot_alignment(raw_edf.info, trans=coreg.trans, **plot_kwargs)

    return coreg.trans["trans"][:-1].ravel()


# Warp locations to standard head surface:
def warp_locs(self, raw):
    """Warp locs.

    Parameters
    ----------
    raw : mne.io.Raw
        an instance of mne.io.Raw

    Returns
    -------
    None (operates in place)
    """
    if "montage_info" in self.config["replace_string"]:
        if isinstance(self.config["replace_string"]["montage_info"], str):
            pass
            # TODO: if it is a BIDS channel tsv, load the tsv,sd_t_f_vals
            # else read the file that is assumed to be a transformation matrix.
        else:
            pass
            # raw = (warp_locs(raw, c01_config['ref_loc_file'],
            # 'transform',[[montage_info]],
            # 'manual','off'))
            # MNE does not apply the transform to the montage permanently.
