# coding: utf-8

"""Clases and Functions for running the Lossless Pipeline."""

from mne.utils import logger
import mne_bids
from mne_bids import get_bids_path_from_fname, BIDSPath
import numpy as np
from pathlib import Path
import tempfile

# BIDS
import mne

# Find breaks
from mne.preprocessing import annotate_break

# Co-Registration
from mne.coreg import Coregistration

# nearest neighbours
import pandas as pd
import xarray as xr
import scipy
from scipy.spatial import distance_matrix
from functools import partial
from tqdm.notebook import tqdm

# ICA
from mne.preprocessing import ICA
import mne_icalabel
from mne_icalabel.annotation import write_components_tsv
from mne_icalabel.config import ICLABEL_LABELS_TO_MNE

from .config import Config

class FlaggedChs(dict):
    """Object for handling flagged channels in an instance of mne.Raw.
    
    Methods
    -------
    add_flag_cat:
        Append a list of one or more channel names that should be considered
        `'bad'` to the `'manual'` `dict` key.
    rereference:
        re-ference instance of mne.Raw, using the mne.Raw.set_eeg_reference
        method. Applicable only for EEG data.
    """

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        super().__init__(*args, **kwargs)
        if 'manual' not in self:
            self['manual'] = []

    def add_flag_cat(self, kind, bad_ch_names):
        """Append a list of channel names to the 'manual' dict key.

        Parameters:
        -----------
            kind : str
                Should be one of 'outlier', 'ch_sd', 'low_r', 'bridge', 'rank'.
            bad_ch_names : list | tuple
                Channel names. Will be the values for the `kind` `dict` `key`.
        """
        self[kind] = bad_ch_names
        self['manual'] = np.unique(np.concatenate(list(self.values())))

    def rereference(self, inst, **kwargs):
        """Re-reference instance of mne.Raw.
        
        Parameters
        ----------
        inst : mne.Raw
            An instance of mne.Raw that contains channels of type `EEG`.
        kwargs : `dict`
            `dict` of valid keyword arguments for the
            `mne.Raw.set_eeg_reference` method.
        """
        inst.set_eeg_reference(ref_channels=[ch for ch in inst.ch_names
                                             if ch not in self['manual']],
                               **kwargs)

    def save_tsv(self, fname):
        labels = []
        ch_names = []
        for key in self:
            labels.extend([key]*len(self[key]))
            ch_names.extend(self[key])
        pd.DataFrame({"labels": labels,
                      "ch_names": ch_names}).to_csv(fname,
                                                    index=False, sep="\t")

    def load_tsv(self, fname):
        df = pd.read_csv(fname, sep='\t')
        for label, grp_df in df.groupby("labels"):
            self[label] = grp_df.ch_names.values


class FlaggedEpochs(dict):
    """Object for handling flagged Epochs in an instance of mne.Epochs.
    
    Methods
    -------
    add_flag_cat:
        Append a list of indices (corresponding to Epochs in an instance of
        mne.Epochs) to the 'manual' `dict` key.
    """

    def __init__(self, *args, **kwargs):
        """Initialize class.
        
        Parameters
        ----------
        args : list | tuple
            positional arguments accepted by `dict` class
        kwargs : dict
            keyword arguments accepted by `dict` class
        """
        super().__init__(*args, **kwargs)
        if 'manual' not in self:
            self['manual'] = []

    def add_flag_cat(self, kind, bad_epoch_inds, raw, epochs):
        """Append a list of channel names to the 'manual' dict key.

        Parameters:
        -----------
        kind : str
            Should be one of 'ch_sd', 'low_r' 'ic_sd1'.
        bad_epochs_inds : list | tuple
            Indices for the epochs in an `mne.Epochs` object. Will be the
            values for the `kind` `dict` `key`.
        raw : mne.raw
            an instance of mne.Raw
        epochs : mne.Epochs
            an instance of mne.Epochs
        """
        self[kind] = bad_epoch_inds
        self['manual'] = np.unique(np.concatenate(list(self.values())))
        add_pylossless_annotations(raw, bad_epoch_inds, kind, epochs)

    def load_from_raw(self, raw):
        sfreq = raw.info['sfreq']
        for annot in raw.annotations:
            if annot['description'].startswith('bad_pylossless'):
                ind_onset = int(np.round(annot['onset'] * sfreq))
                ind_dur = int(np.round(annot['duration'] * sfreq))
                inds = np.arange(ind_onset, ind_onset + ind_dur)
                if annot['description'] not in self:
                    self[annot['description']] = list()
                self[annot['description']].append(inds)


class FlaggedICs(dict):
    """Object for handling IC classification in an instance of mne.ICA.

    Attributes
    ----------
    fname : `pathlib.Path`
        Filepath to the `derivatives/pylosssless` folder in the `bids_root`
        directory.
    ica : `mne.ICA`
        An instance of `mne.ICA` to be passed into `mne.icalabel`
    data_frame : `pd.DataFrame`
        An instance of `pd.DataFrame` that contains the `dict` returned by
        `mne.icalabel.label_components`.
    Methods
    -------
    add_flag_cat :
        Label one or more Independent Components in an instance of mne.ICA,
        with one of the labels from mne.icalabel ('brain', 'channel' etc.).
    label_components :
        Labels the independent components in an instance of mne.ICA using
        mne.icalabel.label_components. Assigns returned `dict` to
        `self.data_frame`
    save :
        Save the ic_labels returned by `mne.icalabel.write_components_tsv` to
        the `derivatives/pylossless` folder in the `bids_root` directory.
    """

    def __init__(self, *args, **kwargs):
        """Initialize class.
        
        Parameters
        ----------
        args : list | tuple
            positional arguments accepted by `dict` class
        kwargs : dict
            keyword arguments accepted by `dict` class.
        """
        super().__init__(*args, **kwargs)
        self.fname = None
        self.data_frame = None

    def label_components(self, epochs, ica):
        """Classify components using mne_icalabel.
        
        Parameters
        ----------
        epochs : mne.Epochs
            instance of `mne.Epochs` to be passed into
            `mne_icalabel.label_components`.
        ica : mne.ICA
            instance of `mne.ICA` to be passed into
            `mne_icalabel.label_components`.
        method : str (default "iclabel")
            The proposed method for labeling components, to be passed into
            `mne_icalabel.label_components`. Must be one of: `'iclabel'`.
        """
        mne_icalabel.label_components(epochs, ica, method="iclabel")
        self.data_frame = _icalabel_to_data_frame(ica)

    def save_tsv(self, fname):
        """Save IC labels.

        Parameters
        ----------
        fname : str | pathlib.Path
            The output filename.
        """
        self.fname = fname
        self.data_frame.to_csv(fname, sep='\t', index=False, na_rep='n/a')

    def load_tsv(self, fname, data_frame=None):
        self.fname = fname
        if data_frame is None:
            data_frame = pd.read_csv(fname, sep='\t')
        self.data_frame = data_frame

def epochs_to_xr(epochs, kind="ch", ica=None):
    """Create an Xarray DataArray from an instance of mne.Epochs.

    Parameters
    ----------
        epochs : mne.Epochs
            an instance of mne.Epochs
        kind : str (default 'ch')
            The name to be passed into the `coords` argument of xr.DataArray
            corresponding to the channel dimension of the epochs object.
            Must be 'ch' or 'ic'.
    Returns
    -------
        An Xarray DataArray object.
    """
    if kind == "ch":
        data = epochs.get_data()  # n_epochs, n_channels, n_times
        data = xr.DataArray(epochs.get_data(),
                            coords={'epoch': np.arange(data.shape[0]),
                                    "ch": epochs.ch_names,
                                    "time": epochs.times})
    elif kind == "ic":
        data = ica.get_sources(epochs).get_data()
        data = xr.DataArray(epochs.get_data(),
                            coords={'epoch': np.arange(data.shape[0]),
                                    "ic": epochs.ch_names,
                                    "time": epochs.times})
    else:
        raise ValueError("The argument kind must be equal to 'ch' or 'ic'.")

    return data


def _icalabel_to_data_frame(ica):
    """ """
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

def get_operate_dim(array, flag_dim):
    """Get the Xarray.DataArray dimension to use with pipeline funcs.
    
    Parameters
    ----------
    array : Xarray DataArray
        An instance of `Xarray.DataArray` that was constructed from an
        `mne.Epochs` object, using `pylossless.pipleine.epochs_to_xr`.
    flag_dim : str
        Name of the Xarray.DataArray.dims to remove. Must be one of 'epoch',
        'ch', or 'ic'.

    Returns
    -------
    list : an instance of `list`
        a `list` containing the `dims` of the passed in `Xarray.DataArray`,
        with the `flag_dim` removed from the list.
    """
    dims = list(array.dims)
    dims.remove(flag_dim)
    return dims


def variability_across_epochs(epochs_xr, var_measure='sd',
                              epochs_inds=None, ch_names=None,
                              ic_inds=None, spect_range=()):
    """Compute variability across epochs.

    Parameters
    ----------
    epochs_xr : `Xarray.DataArray`
        An instance of `Xarray.DataArray` that was constructed from an
        `mne.Epochs` object, using `pylossless.pipleine.epochs_to_xr`.
    var_measure : str (default 'sd')
        The measure to assess variability. Must be one of 'sd' or 'absmean'.
    epochs_inds : list | tuple (default `None`)
        Indices of the epochs that should be included in the variability
        assessment. Indices must correspond existing values in
        `epochs_xr['epoch']`. If `None`, The Epoch is ignored.
    ch_names : list | tuple (default `None`)
        Names of the channels that should be included in the variability
        assessment. Names must corresond to existing values in
        `epochs_xr['ch']`. If `None`, channel name dimension is ignored.
    ic_inds : list | tuple (default `None`)
        Indices of the independent components in epochs_xr['ic'] to be
        included in the variability assessment. Indices must correspond to
        existing values in `epochs_xr['ic']`. Only Valid if 'ic' was
        passed into the `kind` argument of `pylossless.pipeline.epochs_to_xr`.
        If `None`, IC dimension is ignored.
    spect_range : tuple (default empty tuple)
        Not currently implemented.

    Returns
    -------
    Xarray DataArray : Xarray.DataArray
        An instance of Xarray.DataArray, with shape n_channels, by n_times.
    """
    if ch_names is not None:
        epochs_xr = epochs_xr.sel(ch=ch_names)
    if epochs_inds is not None:
        epochs_xr = epochs_xr.sel(epoch=epochs_inds)
    if ic_inds is not None:
        epochs_xr = epochs_xr.sel(ic=ic_inds)

    if var_measure == 'sd':
        return epochs_xr.std(dim="epoch")  # returns array of shape (n_chans, n_times)
    if var_measure == 'absmean':
        return np.abs(epochs_xr).mean(dim="epoch")

    if var_measure == 'spect':

        #        p=abs(fft(bsxfun(@times,data,hanning(EEG.pnts)'),[],2));
        #        fstep=EEG.srate/EEG.pnts;
        #        f=[fstep:fstep:EEG.srate]-fstep;
        #        [val,ind(1)]=min(abs(f-(g.spectrange(1))));
        #        [val,ind(2)]=min(abs(f-(g.spectrange(2))));
        #        data_sd=squeeze(mean(p(:,ind(1):ind(2),:),2));
        raise NotImplementedError


# TODO change naming of 'init' and init_dir specifically,
# neg/pos/both for lower/upper bound options.
def marks_array2flags(inarray, flag_dim='epoch', outlier_method='q',
                      init_vals=(), init_dir='both', init_crit=(),
                      flag_method='z_score', flag_vals=(),
                      flag_crit=(), trim=0):
    """Mark epochs, channels, or ICs as flagged for artefact.

    This function takes an array with typically created by chan_variance or
    chan_neighbour_r and marks either periods of time or sources as outliers.
    Often these discovered time periods are artefactual and are marked as such.
    An array of values representating the distribution of values inside an
    epoch are passed to the function. Next, these values are put through one
    of three outlier detection schemes. Epochs that are outliers are marked
    as 1's and are 0 otherwise in a second data array. This array is then
    averaged column-wise or row-wise. Column-wise averaging results in
    flagging of time, while row-wise results in rejection of sources. This
    averaged distribution is put through another round of outlier detection.
    This time, if data points are outliers, they are flagged.

    Parameters
    ----------
    inarray : Xarray.DataArray

    flag_dim : str
        Must be one of 'epoch', 'ch', 'ic'. Col flags time, row flags
        sources. 
    
    outlier_method : str
        Must be one of 'q', 'z', or 'fixed'.

    init_vals : list | tuple (default empty tuple)
        A list or tuple containing two elements, the lower and upper bound to
        be used as thresholds. if `outlier_method` == `'q'`, the thresholds
        should be quantiles between 0 and 1 e.g. `[.3, .7]`. If
        `outlier_method` == `'fixed'`, the thresholds should be `int` or
        `float`, marking some threshold for exclusion .e.g `[0, 50]`. If
        `outlier_method` == 'z', this parameter is ignored.
    init_dir : str (default 'both')
        Must be one of 'pos', 'neg', 'both'. Allows looking for unusually low
        (neg) correlations, high (pos), or both.
    init_crit : int
        If `outlier_method` == `'q'`, this value scales the distancea long with
        `flag_crit`.
        If `outlier_method` == `'fixed'`, this parameter is ignored.
    flag_method : str
        must be one of 'q', 'z', or 'fixed'. Second pass responsible for
        flagging aggregate array.
    flag_vals :
        Second pass for flagging.. if `outlier_method` == `'q'`, the quantile
        to use, e.g [.7].
    flag_crit : float
        Second pass for flagging.
        If `outler_method` == `'fixed'` and `'fixed'` is selected for
        `flag_method`, This value should bea threshhold that something must
        pass to be flagged. i.e. if 0.2 (20%) of channels behaving as
        outliers. If `outlier_method` == `'q'`, this value scales the distance
        along with `init_crit`.
    trim  : int
        Numerical value of trimmed mean and std. Only valid when
        `outlier_method` == `'z'`.

    Returns
    -------
    outlier_mask : np.array
        Mask of periods of time that are flagged as outliers
    
    outind : np.array
        Array of 1's and 0's. 1's represent flagged sources/time.
        Indices are only flagged if out_dist array fall above a second
        outlier threshhold.
     out_dist : np.array
        Distribution of rejection array. Either the mean row-wise or
        column-wise of outlier_mask.

    Notes
    -----
    Read below for an indepth description of the outlier_methods:
    fixed
    This method does no investigation of the distribution. Instead
    specific criteria are given via `vals` and `crit`. If `'fixed'`
    is selected for the `outlier_method` option, only `init_vals` should
    be filled while `init_crit` is to be left empty. This would have the
    effect of marking some interval, e.g. [0 50] as being an outlier.
    If `'fixed'` is selected for `'flag_method'`, only `'flag_crit'` should be
    filled. This translates to a threshhold that something must pass
    to be flagged. i.e. if 0.2 (20%) of channels are behaving as
    outliers, then the period of time is flagged. Conversely if
    `'flag_dim'` is row, if a source is bad 20% of the time it is marked
    as a bad channel.
    q (Quantile)
    `'init_vals'` allows for the specification of which
    quantiles to use, e.g. [.3 .7]. When using `'flag_vals'`, only specify
    one quantile, e.g [.7]. The absolute difference between the
    median and these quantiles returns a distance. `'init_crit'` and
    `'flag_crit'` scales this distance. If values are found to be outside
    of this, they are flagged.
    z (z-score)
    Typical z-score calculation for distance. Vals and crit options
    not used for this methodology. See trim option above for control.
    """
    # Calculate mean and standard deviation for each column
    operate_dim = get_operate_dim(inarray, flag_dim)
    if outlier_method == 'q':
        if len(init_vals) == 1:
            qval = [.5 - init_vals[0], .5, .5 + init_vals[0]]
        elif len(init_vals) == 2:
            qval = [init_vals[0], .5, init_vals[1]]
        elif len(init_vals) == 3:
            qval = init_vals
        else:
            raise ValueError('init_vals argument must be 1, 2, or 3')

        m_dist = inarray.quantile(qval[1], dim=operate_dim)
        l_dist = inarray.quantile(qval[0], dim=operate_dim)
        u_dist = inarray.quantile(qval[2], dim=operate_dim)
        l_out = m_dist - (m_dist - l_dist) * init_crit
        u_out = m_dist + (u_dist - m_dist) * init_crit

    elif outlier_method == 'z':
        trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
        m_dist = inarray.reduce(trim_mean, dim=operate_dim)

        trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))
        s_dist = inarray.reduce(trim_std, dim=operate_dim)
        l_dist = m_dist - s_dist
        u_dist = m_dist + s_dist
        l_out = m_dist - s_dist * init_crit
        u_out = m_dist + s_dist * init_crit

    elif outlier_method == 'fixed':
        l_out, u_out = init_vals

    # flag outlying values
    outlier_mask = xr.zeros_like(inarray, dtype=bool)

    if init_dir == 'pos' or init_dir == 'both':  # for positive outliers
        outlier_mask = outlier_mask | (inarray > u_out)

    if init_dir == 'neg' or init_dir == 'both':  # for negative outliers
        outlier_mask = outlier_mask | (inarray < l_out)

    # average column of outlier_mask
    dims = get_operate_dim(inarray, flag_dim)
    assert (len(dims) == 1)
    critrow = outlier_mask.mean(dims[0])
    if 'epoch' in dims:
        critrow = outlier_mask.mean('ch')

    # set the flag index threshold (may add quantile option here as well)
    if flag_method == 'fixed':
        rowthresh = flag_crit

    elif flag_method == 'z_score':
        mccritrow = np.mean(critrow)
        sccritrow = np.std(critrow)
        rowthresh = mccritrow + sccritrow * flag_crit

    elif flag_method == 'q':
        qval = [.5, flag_vals]
        mccritrow = np.quantile(critrow, qval[0])
        sccritrow = np.quantile(critrow, qval[1])
        rowthresh = mccritrow + (sccritrow - mccritrow) * flag_crit

    else:
        raise ValueError("flag_method must be flag_method, z_score, or q")

    # get indices of rows beyond threshold
    # outind = np.where(critrow > rowthresh)[0]
    outind = np.where(critrow > rowthresh)[0]

    out_dist = np.array([m_dist, l_dist, u_dist, l_out, u_out])

    return outlier_mask, outind, out_dist


def add_pylossless_annotations(raw, inds, event_type, epochs):
    """Add annotations for flagged epochs.
    
    Parameters
    ----------
    raw : mne.Raw
        an instance of mne.Raw
    inds : list | tuple
        indices corresponding to artefactual epochs
    event_type : str
        One of 'ch_sd', 'low_r', 'ic_sd1'
    epochs : mne.Epochs
        an instance of mne.Epochs
    
    Returns
    -------
    Raw : mne.Raw
        an instance of mne.Raw
    """
    # Concatenate epoched data back to continuous data
    t_onset = epochs.events[inds, 0] / epochs.info['sfreq']
    duration = np.ones_like(t_onset) / epochs.info['sfreq'] * len(epochs.times)
    description = [f'bad_pylossless_{event_type}'] * len(t_onset)
    annotations = mne.Annotations(t_onset, duration, description,
                                  orig_time=raw.annotations.orig_time)
    raw.set_annotations(raw.annotations + annotations)
    return raw


def chan_neighbour_r(epochs, nneigbr, method):
    """Compute nearest Neighbor R.

    Parameters:
    -----------
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
    chan_locs = pd.DataFrame(epochs.get_montage().get_positions()['ch_pos']).T
    chan_dist = pd.DataFrame(distance_matrix(chan_locs, chan_locs),
                             columns=chan_locs.index,
                             index=chan_locs.index)
    rank = chan_dist.rank('columns', ascending=True) - 1
    rank[rank == 0] = np.nan
    nearest_neighbor = pd.DataFrame({ch_name: row.dropna()
                                                 .sort_values()[:nneigbr]
                                                 .index.values
                                     for ch_name, row in rank.iterrows()}).T

    r_list = []
    for name, row in tqdm(list(nearest_neighbor.iterrows())):
        this_ch = epochs.get_data(name)
        nearest_chs = epochs.get_data(list(row.values))
        this_ch_xr = xr.DataArray([this_ch * np.ones_like(nearest_chs)],
                                  dims=['ref_chan', 'epoch',
                                        'channel', 'time'],
                                  coords={'ref_chan': [name]})
        nearest_chs_xr = xr.DataArray([nearest_chs],
                                      dims=['ref_chan', 'epoch',
                                            'channel', 'time'],
                                      coords={'ref_chan': [name]})
        r_list.append(xr.corr(this_ch_xr, nearest_chs_xr, dim=['time']))

    c_neigbr_r = xr.concat(r_list, dim='ref_chan')

    if method == 'max':
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).max(dim='channel')

    elif method == 'mean':
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).mean(dim='channel')

    elif method == 'trimmean':
        trim_mean_10 = partial(scipy.stats.trim_mean, proportiontocut=0.1)
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r)\
                       .reduce(trim_mean_10, dim='channel')

    return m_neigbr_r.rename(ref_chan="ch")


# TODO check that annot type contains all unique flags
def marks_flag_gap(raw, min_gap_ms, included_annot_type=None,
                   out_annot_name='bad_pylossless_gap'):
    """Mark small gaps in time between pylossless annotations.
    
    Parameters
    ----------
    raw : mne.Raw
        An instance of mne.Raw
    min_gap_ms : int
        Time in milleseconds. If the time between two consecutive pylossless
        annotations is less than this value, that time period will be
        annotated.
    included_annot_type : str (Default None)
        Descriptions of the `mne.Annotations` in the `mne.Raw` to be included.
        If `None`, includes ('bad_pylossless_ch_sd', 'bad_pylossless_low_r',
        'bad_pylossless_ic_sd1', 'bad_pylossless_gap').
    out_annot_name : str (default 'bad_pylossless_gap')
        The description for the `mne.Annotation` That is created for any gaps.
    
    Returns
    -------
    Annotations : `mne.Annotations`
        An instance of `mne.Annotations`
    """
    if included_annot_type is None:
        included_annot_type = ('bad_pylossless_ch_sd', 'bad_pylossless_low_r',
                               'bad_pylossless_ic_sd1', 'bad_pylossless_gap')

    if len(raw.annotations) == 0:
        return mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)

    ret_val = np.array([[annot['onset'], annot['duration']]
                        for annot in raw.annotations
                        if annot['description'] in included_annot_type]).T

    if len(ret_val) == 0:
        return mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)

    onsets, durations = ret_val
    offsets = onsets + durations
    gaps = np.array([min(onset - offsets[offsets < onset])
                     if np.sum(offsets < onset) else np.inf
                     for onset in onsets[1:]])
    gap_mask = gaps < min_gap_ms / 1000

    return mne.Annotations(onset=onsets[1:][gap_mask] - gaps[gap_mask],
                           duration=gaps[gap_mask],
                           description=out_annot_name,
                           orig_time=raw.annotations.orig_time)


def coregister(raw_edf, fiducials="estimated",  # get fiducials from fsaverage
               show_coreg=False, verbose=False):
    """Coregister Raw object to `'fsaverage'`.
    
    Parameters
    ----------
    raw_edf : mne.Raw
        an instance of `mne.Raw` to coregister.
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
    plot_kwargs = dict(subject='fsaverage',
                       surfaces="head-dense", dig=True, show_axes=True)

    coreg = Coregistration(raw_edf.info, 'fsaverage', fiducials=fiducials)
    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=verbose)

    if show_coreg:
        mne.viz.plot_alignment(raw_edf.info, trans=coreg.trans, **plot_kwargs)

    return coreg.trans['trans'][:-1].ravel()


# Warp locations to standard head surface:
def warp_locs(self, raw):
    """Warp locs.
    
    Parameters:
    -----------
    raw : mne.Raw
        an instance of mne.Raw 
    """
    if 'montage_info' in self.config['replace_string']:
        if isinstance(self.config['replace_string']['montage_info'], str):
            pass
            # TODO if it is a BIDS channel tsv, load the tsv,sd_t_f_vals
            # else read the file that is assumed to be a transformation matrix.
        else:
            pass
            # raw = (warp_locs(raw, c01_config['ref_loc_file'],
            # 'transform',[[montage_info]],
            # 'manual','off'))
            # MNE does not apply the transform to the montage permanently.


class LosslessPipeline():
    """Class used to handle pipeline parameters."""

    def __init__(self, config_fname=None):
        """Initialize class.
        
        Parameters
        ----------
        config_fname : pathlib.Path
            path to config file specifying the parameters to be used
            in the pipeline.
        """
        self.flagged_chs = FlaggedChs()
        self.flagged_epochs = FlaggedEpochs()
        self.flagged_ics = FlaggedICs()
        self.config_fname = config_fname
        if config_fname:
            self.load_config()
        self.raw = None
        self.ica1 = None
        self.ica2 = None

    def load_config(self):
        """Load the config file."""
        self.config = Config().read(self.config_fname)

    def set_montage(self):
        """Set the montage."""
        analysis_montage = self.config['project']['analysis_montage']
        if analysis_montage == "" and self.raw.get_montage() is not None:
            # No analysis montage has been specified and raw already has
            # a montage. Nothing to do; just return. This can happen
            # with a BIDS dataset automatically loaded with its corresponding
            # montage.
            return

        if analysis_montage in mne.channels.montage.get_builtin_montages():
            # If chanlocs is a string of one the standard MNE montages
            montage = mne.channels.make_standard_montage(analysis_montage)
            self.raw.set_montage(montage,
                            **self.config['project']['set_montage_kwargs'])
        else:  # If the montage is a filepath of a custom montage
            raise ValueError('self.config["project"]["analysis_montage"]'
                             ' should be one of the default MNE montages as'
                             ' specified by'
                             ' mne.channels.get_builtin_montages().')
            # montage = read_custom_montage(chan_locs)

    def get_epochs(self, detrend=None, preload=True):
        """Create mne.Epochs according to user arguments.
        
        Parameters
        ----------
        detrend : int | None (default None)
            If 0 or 1, the data channels (MEG and EEG) will be detrended when
            loaded. 0 is a constant (DC) detrend, 1 is a linear detrend.None is
            no detrending. Note that detrending is performed before baseline
            correction. If no DC offset is preferred (zeroth order detrending),
            either turn off baseline correction, as this may introduce a DC
            shift, or set baseline correction to use the entire time interval
            (will yield equivalent results but be slower).
        preload : bool (default True)
            Load epochs from disk when creating the object or wait before
            accessing each epoch (more memory efficient but can be slower).
        
        Returns
        -------
        Epochs : mne.Epochs
            an instance of mne.Epochs
        """
        # TODO: automatically load detrend/preload description from MNE.
        tmin = self.config['epoching']['epochs_args']['tmin']
        tmax = self.config['epoching']['epochs_args']['tmax']
        overlap = self.config['epoching']['overlap']
        events = mne.make_fixed_length_events(self.raw, duration=tmax-tmin,
                                              overlap=overlap)

        epoching_kwargs = self.config['epoching']['epochs_args']
        if detrend is not None:
            epoching_kwargs['detrend'] = detrend
        epochs = mne.Epochs(self.raw, events=events,
                            preload=preload, **epoching_kwargs)
        epochs = (epochs.pick(picks=None, exclude='bads')
                        .pick(picks=None,
                              exclude=list(self.flagged_chs['manual'])))
        self.flagged_chs.rereference(epochs)

        return epochs

    def run_staging_script(self):
        """Run a staging script if specified in config."""
        # TODO
        if 'staging_script' in self.config:
            staging_script = Path(self.config['staging_script'])
            if staging_script.exists():
                exec(staging_script.open().read())

    def find_breaks(self):
        """Find breaks using `mne.preprocessing.annotate_break`."""
        if 'find_breaks' not in self.config or not self.config['find_breaks']:
            return
        breaks = annotate_break(self.raw, **self.config['find_breaks'])
        self.raw.set_annotations(breaks + self.raw.annotations)

    def flag_outlier_chs(self):
        """Flag outlier Channels."""
        # Window the continuous data
        # logging_log('INFO', 'Windowing the continous data...');
        epochs_xr = epochs_to_xr(self.get_epochs(), kind="ch")

        # Determines comically bad channels,
        # and leaves them out of average rereference
        trim_ch_sd = variability_across_epochs(epochs_xr)
        # std across epochs for each chan; shape (chans, time)

        # Measure how diff the std of 1 channel is with respect
        # to other channels (nonparametric z-score)
        ch_dist = trim_ch_sd - trim_ch_sd.median(dim="ch")
        perc_30 = trim_ch_sd.quantile(0.3, dim="ch")
        perc_70 = trim_ch_sd.quantile(0.7, dim="ch")
        ch_dist /= perc_70 - perc_30  # shape (chans, time)

        mean_ch_dist = ch_dist.mean(dim="time")  # shape (chans)

        # find the median and 30 and 70 percentiles
        # of the mean of the channel distributions
        mdn = np.median(mean_ch_dist)
        deviation = np.diff(np.quantile(mean_ch_dist, [0.3, 0.7]))

        bad_ch_names = mean_ch_dist.ch[mean_ch_dist > mdn+6*deviation]
        self.flagged_chs.add_flag_cat(kind='outliers',
                                      bad_ch_names=bad_ch_names)

        # TODO: Verify: It is unclear this is necessary.
        # get_epochs() is systematically rereferencing and
        # all steps (?) uses the get_epochs() function
        self.flagged_chs.rereference(self.raw)

    def flag_ch_sd(self):
        """Flag channels with outlying standard deviation."""
        epochs = self.get_epochs()
        epochs_xr = epochs_to_xr(epochs, kind="ch")
        data_sd = epochs_xr.std("time")

        # flag epochs for ch_sd
        if 'epoch_ch_sd' in self.config:
            config_epoch = self.config['epoch_ch_sd']
            if 'outlier_method' in config_epoch:
                if config_epoch['outlier_method'] is None:
                    del config_epoch['outlier_method']
                elif config_epoch['outlier_method'] not in ('q', 'z', 'fixed'):
                    raise NotImplementedError
        flag_sd_t_inds = marks_array2flags(data_sd, flag_dim='ch',
                                           **config_epoch)[1]
        self.flagged_epochs.add_flag_cat('ch_sd', flag_sd_t_inds, self.raw, epochs)

        # flag channels for ch_sd
        flag_sd_ch_inds = marks_array2flags(data_sd, flag_dim='epoch',
                                            **self.config['ch_ch_sd'])[1]

        bad_ch_names = epochs_xr.ch[flag_sd_ch_inds]
        self.flagged_chs.add_flag_cat(kind='ch_sd',
                                      bad_ch_names=bad_ch_names)

        # TODO: Verify: It is unclear this is necessary.
        # get_epochs() is systematically rereferencing and
        # all steps (?) uses the get_epochs() function
        self.flagged_chs.rereference(self.raw)

    def get_n_nbr(self):
        """Calculate nearest neighbour correlation for channels."""
        # Calculate nearest neighbout correlation on
        # non-'manual' flagged channels and epochs...
        epochs = self.get_epochs()
        n_nbr_ch = self.config['nearest_neighbors']['n_nbr_ch']
        return chan_neighbour_r(epochs, n_nbr_ch, 'max'), epochs

    def flag_ch_low_r(self):
        """Check neighboring channels for too high or low of a correlation.
        
        Returns
        -------
        data array : `numpy.array`
            an instance of `numpy.array`
        """
        # Calculate nearest neighbout correlation on
        # non-'manual' flagged channels and epochs...
        data_r_ch = self.get_n_nbr()[0]

        # Create the window criteria vector for flagging low_r chan_info...
        flag_r_ch_inds = marks_array2flags(data_r_ch, flag_dim='ch',
                                           init_dir='neg',
                                           **self.config['ch_low_r'])[1]

        # Edit the channel flag info structure
        bad_ch_names = data_r_ch.ch[flag_r_ch_inds].values.tolist()
        self.flagged_chs.add_flag_cat(kind='low_r', bad_ch_names=bad_ch_names)

        return data_r_ch

    def flag_ch_bridge(self, data_r_ch):
        """Flag bridged channels.
        
        Parameters
        ----------
        data_r_ch : `numpy.array`
            an instance of `numpy.array`
        """
        # Uses the correlation of neighboors
        # calculated to flag bridged channels.

        msr = data_r_ch.median("epoch") / data_r_ch.reduce(scipy.stats.iqr, dim="epoch")

        trim = self.config['bridge']['bridge_trim']
        if trim >= 1:
            trim /= 100
        trim /= 2

        trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
        trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))

        z = self.config['bridge']['bridge_z']
        mask = msr > msr.reduce(trim_mean, dim="ch") + z*msr.reduce(trim_std, dim="ch")

        bad_ch_names = data_r_ch.ch.values[mask]
        self.flagged_chs.add_flag_cat(kind='bridge',
                                      bad_ch_names=bad_ch_names)

    def flag_ch_rank(self, data_r_ch):
        """Flag the channel that is the least unique.

        Flags the channel that is the least unique, the channel to remove prior
        to ICA in order to account for the rereference rank deficiency.

        Parameters
        ----------
        data_r_ch : `numpy.array`.
            an instance of `numpy.array`.
        """
        if len(self.flagged_chs['manual']):
            ch_sel = [ch for ch in data_r_ch.ch.values if ch not in self.flagged_chs['manual']]
            data_r_ch = data_r_ch.sel(ch=ch_sel)

        bad_ch_names = [str(data_r_ch.median("epoch").idxmax(dim="ch").to_numpy())]
        self.flagged_chs.add_flag_cat(kind='rank',
                                      bad_ch_names=bad_ch_names)

    def flag_epoch_low_r(self):
        """Flag epochs where too many channels are bridged.
        
        Notes
        -----
        Similarly to the neighbor r calculation done between channels this
        section looks at the correlation, but between all channels and for
        epochs of time. Time segments are flagged for removal.
        """
        # Calculate nearest neighbout correlation on
        # non-'manual' flagged channels and epochs...
        data_r_ch, epochs = self.get_n_nbr()

        flag_r_t_inds = marks_array2flags(data_r_ch, flag_dim='epoch',
                                          init_dir='neg',
                                          **self.config['epoch_low_r'])[1]

        self.flagged_epochs.add_flag_cat('low_r',
                                         flag_r_t_inds,
                                         self.raw, epochs)

    def flag_epoch_gap(self):
        """Flag small time periods between pylossless annotations."""
        annots = marks_flag_gap(self.raw, self.config['epoch_gap']['min_gap_ms'])
        self.raw.set_annotations(self.raw.annotations + annots)

    def run_ica(self, run):
        """Run ICA.
        
        Parameters
        ----------
        run : str
            Must be 'run1' or 'run2'. 'run1' is the intial ICA use to flag
            epochs, 'run2' is the final ICA used to classify components with
            `mne_icalabel`.
        """
        ica_kwargs = self.config['ica']['ica_args'][run]
        if 'max_iter' not in ica_kwargs:
            ica_kwargs['max_iter'] = 'auto'
        if 'random_state' not in ica_kwargs:
            ica_kwargs['random_state'] = 97

        epochs = self.get_epochs()
        if run == 'run1':
            self.ica1 = ICA(**ica_kwargs)
            self.ica1.fit(epochs)

        elif run == 'run2':
            self.ica2 = ICA(**ica_kwargs)
            self.ica2.fit(epochs)
            self.flagged_ics.label_components(epochs, self.ica2)
        else:
            raise ValueError("The `run` argument must be 'run1' or 'run2'")

    def flag_epoch_ic_sd1(self):
        """Calculate the IC standard Deviation by epoch window.
        
        Flags windows with too much standard deviation.
        """
        # Calculate IC sd by window
        epochs = self.get_epochs()
        epochs_xr = epochs_to_xr(epochs, kind="ic", ica=self.ica1)
        epoch_ic_sd1 = variability_across_epochs(epochs_xr)

        # Create the windowing sd criteria
        kwargs = self.config['ica']['ic_ic_sd']
        flag_epoch_ic_inds = marks_array2flags(epoch_ic_sd1,
                                               flag_dim='ic', **kwargs)[1]

        self.flagged_epochs.add_flag_cat('ic_sd1', flag_epoch_ic_inds,
                                         self.raw, epochs)

        # icsd_epoch_flags=padflags(raw, icsd_epoch_flags,1,'value',.5);

    def save(self, derivatives_path, overwrite=False):
        """Save the file at the end of the pipeline.
        
        Parameters
        ----------
        derivatives_path : mne_bids.BIDSPath
            path of the derivatives folder to save the file to.
        overwrite : bool (default False)
            whether to overwrite existing files with the same name.
        """
        mne_bids.write_raw_bids(self.raw,
                                derivatives_path,
                                overwrite=overwrite,
                                format='EDF',
                                allow_preload=True)
        #  TODO address derivatives support in MNE bids.
        # use shutils ( or pathlib?) to rename file with ll suffix

        # Save ICAs
        bpath = derivatives_path.copy()
        for this_ica, self_ica, in zip(['ica1', 'ica2'],
                                       [self.ica1, self.ica2]):
            suffix = this_ica + '_ica'
            ica_bidspath = bpath.update(extension='.fif',
                                        suffix=suffix,
                                        check=False)
            self_ica.save(ica_bidspath, overwrite=overwrite)

        # Save IC labels
        iclabels_bidspath = bpath.update(extension='.tsv',
                                         suffix='iclabels',
                                         check=False)
        self.flagged_ics.save_tsv(iclabels_bidspath)
        # TODO epoch marks and ica marks are not currently saved into annots
        # raw.save(derivatives_path, overwrite=True, split_naming='bids')
        config_bidspath = bpath.update(extension='.yaml',
                                       suffix='ll_config',
                                       check=False)
        self.config.save(config_bidspath)

        # Save flagged_chs
        flagged_chs_fpath = bpath.update(extension='.tsv',
                                         suffix='ll_FlaggedChs',
                                         check=False)
        self.flagged_chs.save_tsv(flagged_chs_fpath.fpath.name)

    def filter(self):
        # 5.a. Filter lowpass/highpass
        self.raw.filter(**self.config['filtering']['filter_args'])

        if 'notch_filter_args' in self.config['filtering']:
            notch_args = self.config['filtering']['notch_filter_args']
            # in raw.notch_filter, freqs=None is ok if method=spectrum_fit
            if notch_args['freqs'] is None and 'method' not in notch_args:
                logger.debug('No notch filter arguments provided. Skipping')
            else:
                self.raw.notch_filter(**notch_args)

        # 5.b. Filter notch
        notch_args = self.config['filtering']['notch_filter_args']
        spectrum_fit_method = ('method' in notch_args and
                               notch_args['method'] == 'spectrum_fit')
        if notch_args['freqs'] or spectrum_fit_method:
            # in raw.notch_filter, freqs=None is ok if method=='spectrum_fit'
            self.raw.notch_filter(**notch_args)
        else:
            logger.info('No notch filter arguments provided. Skipping')


    def run(self, bids_path, save=True, overwrite=False):
        """Run the pylossless pipeline.
        
        Parameters
        ----------
        bids_path : `pathlib.Path`
            Path of the individual file `bids_root`.
        save : bool (default True).
            Whether to save the files after completing the pipeline. Defaults
            to `True`. if `False`, files are not saved.
        overwrite : bool (default False).
            Whether to overwrite existing files of the same name.
        """
        self.bids_path = bids_path
        self.raw = mne_bids.read_raw_bids(self.bids_path)
        self.raw.load_data()
        self._run()

        if save:
            self.save(self.get_derivative_path(bids_path), overwrite=overwrite)

    def run_with_raw(self, raw):
        self.raw = raw
        self._run()
        return self.raw

    def _run(self):
        self.set_montage()

        # 1. Execute the staging script if specified.
        self.run_staging_script()

        # find breaks
        self.find_breaks()

        # 2. Determine comically bad channels,
        # and leave them out of average reference
        self.flag_outlier_chs()

        # 3-4.flag epochs and channels based on large Channel Stdev.
        self.flag_ch_sd()

        # 5. Filtering
        self.filter()

        # 6. calculate nearest neighbort r values
        data_r_ch = self.flag_ch_low_r()

        # 7. Identify bridged channels
        self.flag_ch_bridge(data_r_ch)
        # TODO: Check why we don not rerefence after this step.

        # 8. Flag rank channels
        self.flag_ch_rank(data_r_ch)
        # TODO: Verify: It is unclear this is necessary.
        # get_epochs() is systematically rereferencing and
        # all steps (?) uses the get_epochs() function
        self.flagged_chs.rereference(self.raw)

        # 9. Calculate nearest neighbour R values for epochs
        self.flag_epoch_low_r()

        # 10. Flag very small time periods between flagged time
        self.flag_epoch_gap()

        # 11. Run ICA
        self.run_ica('run1')

        # 12. Calculate IC SD
        self.flag_epoch_ic_sd1()

        # 13. TODO integrate labels from IClabels to self.flagged_ics
        self.run_ica('run2')

        # 14. Flag very small time periods between flagged time
        self.flag_epoch_gap()

    def run_dataset(self, paths):
        """Run a full dataset.
        
        Parameters
        ----------
        paths : list | tuple
            a list of the bids_paths for all recordings in the dataset that
            should be run.
        """
        for path in paths:
            self.run(path)

    def load_ll_derivative(self, derivatives_path):
        """ """
        if not isinstance(derivatives_path, BIDSPath):
            derivatives_path = get_bids_path_from_fname(derivatives_path)
        self.raw = mne_bids.read_raw_bids(derivatives_path)
        bpath = derivatives_path.copy()
        # Load ICAs
        for this_ica in ['ica1', 'ica2']:
            suffix = this_ica + '_ica'
            ica_bidspath = bpath.update(extension='.fif', suffix=suffix,
                                        check=False)
            setattr(self, this_ica,
                    mne.preprocessing.read_ica(ica_bidspath.fpath))

        # Load IC labels
        iclabels_bidspath = bpath.update(extension='.tsv', suffix='iclabels',
                                         check=False)
        self.flagged_ics.load_tsv(iclabels_bidspath.fpath)

        self.config_fname = bpath.update(extension='.yaml', suffix='ll_config',
                                         check=False)
        self.load_config()

        # Load Flagged Chs
        flagged_chs_fpath = bpath.update(extension='.tsv',
                                         suffix='ll_FlaggedChs',
                                         check=False)
        self.flagged_chs.load_tsv(flagged_chs_fpath.fpath.name)

        # Load Flagged Epochs
        self.flagged_epochs.load_from_raw(self.raw)

        return self

    def get_derivative_path(self, bids_path, derivative_name='pylossless'):
        lossless_suffix = bids_path.suffix if bids_path.suffix else ""
        lossless_suffix += '_ll'
        lossless_root = bids_path.root / 'derivatives' / derivative_name
        return bids_path.copy().update(suffix=lossless_suffix,
                                       root=lossless_root,
                                       check=False)
