# coding: utf-8
from mne.utils import logger
import mne_bids
import numpy as np
from pathlib import Path

# BIDS
import mne

# Co-Registration
from mne.channels.montage import read_custom_montage
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
from mne_icalabel import label_components
from mne_icalabel.annotation import write_components_tsv

from .config import Config

class FlaggedChs(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'manual' not in self:
            self['manual'] = []

    def add_flag_cat(self, kind, bad_ch_names):
        self[kind] = bad_ch_names
        self['manual'] = np.unique(np.concatenate(list(self.values())))

    def rereference(self, inst, **kwargs):
        inst.set_eeg_reference(ref_channels=[ch for ch in inst.ch_names
                                             if ch not in self['manual']],
                               **kwargs)


class FlaggedEpochs(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'manual' not in self:
            self['manual'] = []

    def add_flag_cat(self, kind, bad_epoch_inds, raw, epochs):
        self[kind] = bad_epoch_inds
        self['manual'] = np.unique(np.concatenate(list(self.values())))
        add_pylossless_annotations(raw, bad_epoch_inds, kind, epochs)


class FlaggedICs(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'manual' not in self:
            self['manual'] = []

    def add_flag_cat(self, kind, bad_epoch_inds, raw, epochs):
        self[kind] = bad_epoch_inds
        self['manual'] = np.unique(np.concatenate(list(self.values())))


def chan_variance(epochs, var_measure='sd', epochs_inds=None, ch_names=None,
                  kind='eeg', ica=None, spect_range=()):
    if ch_names is None:
        ch_names = epochs.ch_names

    if kind == 'ica':
        if ica is not None:
            data = ica.get_sources(epochs).get_data()

    elif kind == 'eeg':
        epochs = epochs.copy()
        if epochs_inds is not None:
            epochs = epochs[epochs_inds]
        epochs.load_data()
        epochs.pick(ch_names)
        data = epochs.get_data()

    else:
        raise NotImplementedError

    if var_measure == 'sd':
        return data.std(axis=0)  # returns array of shape (n_chans, n_times)
    if var_measure == 'absmean':
        return np.abs(data).mean(axis=0)

    if var_measure == 'spect':

        #        p=abs(fft(bsxfun(@times,data,hanning(EEG.pnts)'),[],2));
        #        fstep=EEG.srate/EEG.pnts;
        #        f=[fstep:fstep:EEG.srate]-fstep;
        #        [val,ind(1)]=min(abs(f-(g.spectrange(1))));
        #        [val,ind(2)]=min(abs(f-(g.spectrange(2))));
        #        data_sd=squeeze(mean(p(:,ind(1):ind(2),:),2));
        raise NotImplementedError
        return


# TODO change naming of 'init' and init_dir specifically,
# neg/pos/both for lower/upper bound options.
def marks_array2flags(inarray, flag_dim='epoch', init_method='q', init_vals=(),
                      init_dir='both', init_crit=(), flag_method='z_score',
                      flag_vals=(), flag_crit=(), trim=0):

    ''' This function takes an array typically created by chan_variance or
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

     Output:
     outlier_mask - Mask of periods of time that are flagged as outliers
     outind   - Array of 1's and 0's. 1's represent flagged sources/time.
                Indices are only flagged if out_dist array fall above a second
                outlier threshhold.
     out_dist - Distribution of rejection array. Either the mean row-wise or
                column-wise of outlier_mask.

     Input:
     inarray - Data array created by eiether chan_variance or chan_neighbour_r

    % Varargs:
    % init_dir     - String; one of: 'pos', 'neg', 'both'. Allows looking for
    %                unusually low (neg) correlations, high (pos) or both.
    % flag_dim     - String; one of: 'col', 'row'. Col flags time, row flags
    %                sources.
    % init_method  - String; one of: 'q', 'z', 'fixed'. See method section.
    % init_vals    - See method section.
    % init_crit    - See method section.
    % flag_method  - String; one of: 'q', 'z', 'fixed'. See method section.
    %                Second pass responsible for flagging aggregate array.
    % flag_vals    - See method section. Second pass for flagging.
    % flag_crit    - See method section. Second pass for flagging.
    % trim         - Numerical value of trimmed mean and std. Only valid for z.
    %
    % Methods:
    % fixed - This method does no investigation if the distribution. Instead
    %         specific criteria are given via vals and crit. If fixed
    %         is selected for the init_method option, only init_vals should be
    %         filled while init_crit is to be left empty. This would have the
    %         effect of marking some interval, e.g. [0 50] as being an outlier.
    %         If fixed is selected for flag_method, only flag_crit should be
    %         filled. This translates to a threshhold that something must pass
    %         to be flagged. i.e. if 0.2 (20%) of channels are behaving as
    %         outliers, then the period of time is flagged. Conversely if
    %         flag_dim is row, if a source is bad 20% of the time it is marked
    %         as a bad channel.
    %
    % q     - Quantile method. init_vals allows for the specification of which
    %         quantiles to use, e.g. [.3 .7]. When using flag_vals only specify
    %         one quantile, e.g [.7]. The absolute difference between the
    %         median and these quantiles returns a distance. init_crit and
    %         flag_crit scales this distance. If values are found to be outside
    %         of this, they are flagged.
    %
    % z     - Typical Z score calculation for distance. Vals and crit options
    %         not used for this methodology. See trim option above for control.
    '''

    # if flagdir is column wise rotate the inarray.
    if flag_dim == 'ch':
        inarray = inarray.T

    # return flags indices (outind) from input measure (inarray)

    # Calculate mean and standard deviation for each column
    if init_method == 'q':

        if len(init_vals) == 1:
            qval = [.5 - init_vals[0], .5, .5 + init_vals[0]]
        elif len(init_vals) == 2:
            qval = [init_vals[0], .5, init_vals[1]]
        elif len(init_vals) == 3:
            qval = init_vals
        else:
            raise ValueError('init_vals argument must be 1, 2, or 3')

        m_dist = np.percentile(inarray, qval[1]*100, axis=0)
        l_dist = np.percentile(inarray, qval[0]*100, axis=0)
        u_dist = np.percentile(inarray, qval[2]*100, axis=0)
        l_out = m_dist - (m_dist - l_dist) * init_crit
        u_out = m_dist + (u_dist - m_dist) * init_crit

    elif init_method == 'z':

        m_dist = scipy.stats.mstats.trimmed_mean(inarray, [trim, 1-trim],
                                                 axis=0)
        s_dist = scipy.stats.mstats.trimmed_std(inarray,  [trim, 1-trim],
                                                axis=0)
        l_dist = m_dist - s_dist
        u_dist = m_dist + s_dist
        l_out = m_dist - s_dist * init_crit
        u_out = m_dist + s_dist * init_crit

    elif init_method == 'fixed':
        l_out, u_out = init_vals

    # flag outlying values
    outlier_mask = np.zeros_like(inarray, dtype=bool)

    if init_dir == 'pos' or init_dir == 'both':  # for positive outliers
        outlier_mask = outlier_mask | (inarray > u_out)

    if init_dir == 'neg' or init_dir == 'both':  # for negative outliers
        outlier_mask = outlier_mask | (inarray < l_out)

    # average column of outlier_mask
    critrow = outlier_mask.mean(1)

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

    # if flagdir is column wise rotate the outlier_mask and ouind.
    if flag_dim == 'ch':
        inarray = inarray.T
        outlier_mask = outlier_mask.T
        outind = outind.T

    out_dist = np.array([m_dist, l_dist, u_dist, l_out, u_out])

    return outlier_mask, outind, out_dist


def add_pylossless_annotations(raw, inds, event_type, epochs):
    # Concatenate epoched data back to continuous data
    t_onset = epochs.events[inds, 0] / epochs.info['sfreq']
    duration = np.ones_like(t_onset) / epochs.info['sfreq'] * len(epochs.times)
    description = [f'bad_pylossless_{event_type}'] * len(t_onset)
    annotations = mne.Annotations(t_onset, duration, description,
                                  orig_time=raw.annotations.orig_time)
    raw.set_annotations(raw.annotations + annotations)
    return raw


def chan_neighbour_r(epochs, nneigbr, method):

    ''' This function computes the correlation matricies to be passed to
     markup functions to either reject periods of time or individual sources.

     Output:
     EEG         - Standard EEG structure
     m_neighbr_r - Data array to pass to flagging function
     chandist    - Matrix storing distance between channels
     y           - Sorted list of channel distances
     chan_win_sd - Std of EEG.data for quick use (not really necessary)

     Input:
     EEG     - Standard EEG structure
     nneigbr - Number of neighbours to compare in open interval
     method  - String; one of: 'max', 'mean', 'trimmean'. This is the function
               which aggregates the neighbours into one value.

     chan_inds  - Array of 1's and 0's marking which channels to consider.
                  See epoch_inds.
     epoch_inds - Array of 1's and 0's marking epochs to consider. Typically
                  this array is created via marks_label2index.
     '''

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
        trim_mean_10 = partial(scipy.stats.trim_mean,
                               proportiontocut=0.1, axis=0)
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r)\
                              .reduce(trim_mean_10, dim='channel')

    return m_neigbr_r.transpose("epoch", "ref_chan")


def ve_trimmean(data, ptrim, axis=0):
    return ve_trim_var(data, ptrim, np.mean, axis)


def ve_trimstd(data, ptrim, axis=0):
    return ve_trim_var(data, ptrim, np.std, axis)


def ve_trim_var(data, ptrim, func, axis=0):
    if ptrim >= 1:
        ptrim /= 100
    ptrim /= 2

    # Take 1/2 of the requested amount off the top and off the bottom
    data_srt = np.sort(data, axis=axis)
    ntrim = np.round(data.shape[axis]*ptrim)
    indices = np.arange(ntrim, data.shape[axis]-ntrim).astype(int)
    if axis:
        return func(data_srt[:, indices], axis=axis)
    return func(data_srt[indices, :], axis=axis)


# TODO check that annot type contains all unique flags
def marks_flag_gap(raw, min_gap_ms, included_annot_type=None,
                   out_annot_name='bad_pylossless_gap'):
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

    def __init__(self, config_fname):
        self.flagged_chs = FlaggedChs()
        self.flagged_epochs = FlaggedEpochs()
        self.flagged_ics = FlaggedICs()
        self.config_fname = config_fname
        self.load_config()
        #self.init_variables = read_config(init_fname)
        #init_path = Path(self.config['out_path']) / self.config["project"]['id']
        #init_path.mkdir(parents=True, exist_ok=True)
        self.ica1 = None
        self.ica2 = None
        self.ic_labels = None

    def load_config(self):
        self.config = Config(self.config_fname).read()

    def set_montage(self, raw):
        analysis_montage = self.config['project']['analysis_montage']
        if analysis_montage == "" and raw.get_montage() is not None:
            # No analysis montage has been specified and raw already has
            # a montage. Nothing to do; just return. This can happen
            # with a BIDS dataset automatically loaded with its corresponding
            # montage.
            return

        if analysis_montage in mne.channels.montage.get_builtin_montages():
            # If chanlocs is a string of one the standard MNE montages
            montage = mne.channels.make_standard_montage(analysis_montage)
            raw.set_montage(montage, **self.config['project']['set_montage_kwargs'])
        else:  # If the montage is a filepath of a custom montage
            raise ValueError('self.config["project"]["analysis_montage"]'
                             ' should be one of the default MNE montages as'
                             ' specified by mne.channels.get_builtin_montages().')
            # montage = read_custom_montage(chan_locs)

    def get_epochs(self, raw, detrend=None, preload=True):
        epoching_kwargs = self.config['epoching']['epochs_args']
        if detrend is not None:
            epoching_kwargs['detrend'] = detrend
        step = self.config['epoching']['recur_sec'] * raw.info['sfreq']
        first_col = np.arange(0, len(raw.times), step).astype(int)
        events = np.array([first_col,
                           np.zeros_like(first_col),
                           np.zeros_like(first_col)]).T
        epochs = mne.Epochs(raw, events=events,
                            preload=preload, **epoching_kwargs)
        epochs = (epochs.pick(picks=None, exclude='bads')
                        .pick(picks=None,
                              exclude=list(self.flagged_chs['manual'])))
        self.flagged_chs.rereference(epochs)

        return epochs

    def run_staging_script(self):
        # TODO
        if 'staging_script' in self.config:
            staging_script = Path(self.config['staging_script'])
            if staging_script.exists():
                exec(staging_script.open().read())

    def flag_outlier_chs(self, raw):
        # Window the continuous data
        # logging_log('INFO', 'Windowing the continous data...');
        epochs = self.get_epochs(raw)

        # Determines comically bad channels,
        # and leaves them out of average rereference
        trim_ch_sd = chan_variance(epochs)  # std across epochs for each chan

        # Measure how diff the std of 1 channel is with respect
        # to other channels (nonparametric z-score)
        ch_dist = (trim_ch_sd - np.median(trim_ch_sd, axis=0))
        ch_dist /= np.diff(np.percentile(trim_ch_sd, [30, 70], axis=0), axis=0)

        mean_ch_dist = np.mean(ch_dist, axis=1)

        # find the median and 30 and 70 percentiles
        # of the mean of the channel distributions
        mdn = np.median(mean_ch_dist)
        deviation = np.diff(np.percentile(mean_ch_dist, [30, 70]))

        mask = mean_ch_dist > mdn+6*deviation
        bad_ch_names = np.array(epochs.ch_names)[mask]
        self.flagged_chs.add_flag_cat(kind='outliers',
                                      bad_ch_names=bad_ch_names)
        self.flagged_chs.rereference(raw)

    def flag_ch_sd(self, raw):

        epochs = self.get_epochs(raw)
        data_sd = epochs.get_data().std(axis=-1)

        # flag epochs for ch_sd
        if self.config['epoch_ch_sd']['init_method'] is None:
            flag_sd_t_inds = []

        elif self.config['epoch_ch_sd']['init_method'] == 'q':
            kwargs = self.config['epochs_ch_sd']
            flag_sd_t_inds = marks_array2flags(data_sd, flag_dim='epoch',
                                               **kwargs)[1]
        else:
            raise NotImplementedError

        self.flagged_epochs.add_flag_cat('ch_sd', flag_sd_t_inds, raw, epochs)

        # flag channels for ch_sd
        flag_sd_ch_inds = marks_array2flags(data_sd, flag_dim='ch',
                                            **self.config['ch_ch_sd'])[1]

        bad_ch_names = np.array(epochs.ch_names)[flag_sd_ch_inds]
        self.flagged_chs.add_flag_cat(kind='ch_sd',
                                      bad_ch_names=bad_ch_names)
        self.flagged_chs.rereference(raw)

    def flag_ch_low_r(self, raw):
        ''' Checks neighboring channels for
            too high or low of a correlation.'''

        # Calculate nearest neighbout correlation on
        # non-'manual' flagged channels and epochs...
        epochs = self.get_epochs(raw)

        n_nbr_ch = self.config['nearest_neighbors']['n_nbr_ch']
        data_r_ch = chan_neighbour_r(epochs, n_nbr_ch, 'max')

        # Create the window criteria vector for flagging low_r chan_info...
        flag_r_ch_inds = marks_array2flags(data_r_ch.values.T, flag_dim='ch',
                                           init_dir='neg',
                                           **self.config['ch_low_r'])[1]

        # Edit the channel flag info structure
        bad_ch_names = data_r_ch.ref_chan[flag_r_ch_inds].values.tolist()
        self.flagged_chs.add_flag_cat(kind='low_r', bad_ch_names=bad_ch_names)

        return data_r_ch

    def flag_ch_bridge(self, raw, data_r_ch):
        # Uses the correlation of neighboors
        # calculated to flag bridged channels.

        msr = np.median(data_r_ch, 0) / scipy.stats.iqr(data_r_ch, 0)

        mask = (msr > (ve_trimmean(msr[:, None],
                                   self.config['bridge']['bridge_trim'], 0) +
                       ve_trimstd(msr[:, None],
                                  self.config['bridge']['bridge_trim'], 0) *
                       self.config['bridge']['bridge_z'])
                )

        bad_ch_names = data_r_ch.ref_chan.values[mask]
        self.flagged_chs.add_flag_cat(kind='bridge',
                                      bad_ch_names=bad_ch_names)

    def flag_ch_rank(self, raw, data_r_ch, pick_types='eeg'):
        '''Flags the channel that is the least unique,
        the channel to remove prior to ICA in
        order to account for the rereference rank deficiency.'''

        epochs = self.get_epochs(raw).pick(pick_types)
        x = data_r_ch.sel(ref_chan=epochs.ch_names)
        inds = x.argmax(dim=["epoch", "ref_chan"])["ref_chan"]
        bad_ch_names = [str(x.ref_chan[inds].values)]
        self.flagged_chs.add_flag_cat(kind='rank',
                                      bad_ch_names=bad_ch_names)

        self.flagged_chs.rereference(raw)

    def flag_epoch_low_r(self, raw, data_r_ch):
        ''' Similarly to the neighbor r calculation
         done between channels this section looks at the correlation,
         but between all channels and for epochs of time.
         Time segments are flagged for removal.'''

        epochs = self.get_epochs(raw)
        n_nbr_epoch = self.config['nearest_neighbors']['n_nbr_epoch']
        data_r_ch = chan_neighbour_r(epochs, n_nbr_epoch, 'max')

        flag_r_t_inds = marks_array2flags(data_r_ch.values.T, flag_dim='epoch',
                                          init_dir='neg',
                                          **self.config['epoch_low_r'])[1]

        self.flagged_epochs.add_flag_cat('low_r', flag_r_t_inds, raw, epochs)

    def flag_epoch_gap(self, raw):
        annots = marks_flag_gap(raw, self.config['epoch_gap']['min_gap_ms'])
        raw.set_annotations(raw.annotations + annots)

    def run_ica(self, raw, run):
        ica_kwargs = self.config['ica']['ica_args'][run]
        if 'max_iter' not in ica_kwargs:
            ica_kwargs['max_iter'] = 'auto'
        if 'random_state' not in ica_kwargs:
            ica_kwargs['random_state'] = 97

        epochs = self.get_epochs(raw)
        if run == 'run1':
            self.ica1 = ICA(**ica_kwargs)
            self.ica1.fit(epochs)

        elif run == 'run2':
            self.ica2 = ICA(**ica_kwargs)
            self.ica2.fit(epochs)
            self.ic_labels = label_components(epochs, self.ica2,
                                              method="iclabel")
        else:
            raise ValueError("The `run` argument must be 'run1' or 'run2'")

    def flag_epoch_ic_sd1(self, raw):
        '''Calculates the IC standard Deviation by epoch window. Flags windows with
           too much standard deviation.'''

        # Calculate IC sd by window
        epochs = self.get_epochs(raw)
        epoch_ic_sd1 = chan_variance(epochs, kind='ica', ica=self.ica1)

        # Create the windowing sd criteria
        kwargs = self.config['ica']['ic_ic_sd']
        flag_epoch_ic_inds = marks_array2flags(epoch_ic_sd1.T,
                                               flag_dim='epoch', **kwargs)[1]

        self.flagged_epochs.add_flag_cat('ic_sd1', flag_epoch_ic_inds,
                                         raw, epochs)

        # icsd_epoch_flags=padflags(raw, icsd_epoch_flags,1,'value',.5);

    def save(self, raw, bids_path):
        lossless_suffix = bids_path.suffix + '_ll'
        lossless_root = bids_path.root / 'derivatives' / 'pylossless'
        derivatives_path = bids_path.copy().update(suffix=lossless_suffix,
                                                   root=lossless_root,
                                                   check=False
                                                   )
        mne_bids.write_raw_bids(raw,
                                derivatives_path,
                                overwrite=True,
                                format='EDF',
                                allow_preload=True)
                                #  TODO address derivatives support in MNE bids.
                                # use shutils ( or pathlib?) to rename file with ll suffix

        # Save ICAs
        for this_ica, self_ica, in zip(['ica1', 'ica2'],
                                       [self.ica1, self.ica2]):
            ica_bidspath = derivatives_path.copy().update(extension='.fif',
                                                          suffix=this_ica,
                                                          check=False)
            self_ica.save(ica_bidspath)

        # Save IC labels
        iclabels_bidspath = derivatives_path.copy().update(extension='.tsv',
                                                           suffix='iclabels',
                                                           check=False)
        write_components_tsv(self.ica2, iclabels_bidspath)


        # TODO epoch marks and ica marks are not currently saved into annotations
        #raw.save(derivatives_path, overwrite=True, split_naming='bids')

    def run(self, bids_path, save=True):
        raw = mne_bids.read_raw_bids(bids_path)
        raw.load_data()
        self.set_montage(raw)

        # Execute the staging script if specified.
        self.run_staging_script()

        # Determine comically bad channels,
        # and leave them out of average reference
        self.flag_outlier_chs(raw)

        # flag epochs and channels based on large Channel Stdev.
        self.flag_ch_sd(raw)

        # Filter
        raw.filter(**self.config['filtering']['filter_args'])

        if 'notch_filter_args' in self.config['filtering']:
            notch_args = self.config['filtering']['notch_filter_args']
            # in raw.notch_filter, freqs=None is ok if method=spectrum_fit
            if notch_args['freqs'] is None and 'method' not in notch_args:
                logger.debug('No notch filter arguments provided. Skipping')
            else:
                raw.notch_filter(**notch_args)

        # calculate nearest neighbort r values
        data_r_ch = self.flag_ch_low_r(raw)

        # Identify bridged channels
        self.flag_ch_bridge(raw, data_r_ch)

        # FLAG RANK CHAN
        self.flag_ch_rank(raw, data_r_ch)

        # Calculate nearest neighbour R values for epochs
        self.flag_epoch_low_r(raw, data_r_ch)

        # flag very small time periods between flagged time
        self.flag_epoch_gap(raw)

        # Run ICA
        self.run_ica(raw, 'run1')

        # Calculate IC SD
        self.flag_epoch_ic_sd1(raw)

        # TODO 2ND ICA excluding flagged component times
        self.run_ica(raw, 'run2')

        self.flag_epoch_gap(raw)

        if save:
            self.save(raw, bids_path)

    def run_dataset(self, paths):
        for path in paths:
            self.run(path)


"""def pad_flags(raw, flags, npad, varargin):
    ''' Function which given an array 'flags' of values,
     prepends and appends a value around a given nonzero block of data
     in the given array. This value can be customized via the vararg 'value'.
     (e.g. 'value',0.5)'''


    for np=1:npad:
        for i=1:size(flags,3)-1:
            if any(flags(:,:,i+1)) && ~any(flags(:,:,i)):
                flags(:,:,i)=g.value
            if any(flags(:,:,(EEG.trials-(i-1))-1)) && ~any(flags(:,:,EEG.trials-(i-1))):
                flags(:,:,EEG.trials-(i-1))=g.value"""
