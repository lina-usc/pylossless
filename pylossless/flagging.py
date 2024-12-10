# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Classes to store information on artifactual channels, epochs, components."""

import numpy as np
import pandas as pd
import xarray as xr

from mne.utils import logger

import mne_icalabel

from .utils._utils import _icalabel_to_data_frame

IC_LABELS = mne_icalabel.config.ICA_LABELS_TO_MNE
CH_LABELS: list = ["noisy", "bridged", "uncorrelated", "rank"]
EPOCH_LABELS: list = ["noisy", "noisy_ICs", "uncorrelated"]


class _Flagged(dict):

    def __init__(self, keys, kind_str, ll, *args, **kwargs):
        """Initialize class."""
        super().__init__(*args, **kwargs)
        self.ll = ll
        self._keys = keys
        self._kind_str = kind_str

    @property
    def valid_keys(self):
        """Return the valid keys."""
        return tuple(self._keys)

    def __repr__(self):
        """Return a string representation."""
        ret_str = f"Flagged {self._kind_str}s: |\n"
        for key in self._keys:
            ret_str += f"  {key.title().replace('_', ' ')}: {self.get(key, None)}\n"
        return ret_str

    def __eq__(self, other):
        for key in self.valid_keys:
            if not np.array_equal(self.get(key, np.array([])),
                                  other.get(key, np.array([]))):
                return False
        return True

    def __ne__(self, other):
        return not self == other


class FlaggedChs(_Flagged):
    """Object for handling flagged channels in an instance of mne.io.Raw.

    Attributes
    ----------
    ll : LosslessPipeline
        the :class:`~pylossless.pipeline.LosslessPipeline` object that is flagging
        artifactual channels.

    Methods
    -------
    add_flag_cat:
        Add a list of one or more channel names that should be considered
        as artifactual.
    rereference:
        rereference instance of :class:`mne.io.Raw`, using the
        :meth:`mne.io.Raw.set_eeg_reference` method. Applicable only for EEG
        data.
    save_tsv:
        Save flagged channel annotations to a text file.
    load_tsv:
        Load previously saved channel annotations from a text file.

    Notes
    -----
    This class inherits from :class:`dict`, and can use any valid attributes
    and methods for python dictionaries.
    """

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        super().__init__(CH_LABELS, "channel", *args, **kwargs)

    def add_flag_cat(self, kind, bad_ch_names, *args):
        """Store channel names that have been flagged by pipeline.

        Parameters
        ----------
        kind : str
            Should be one of the values in ``CH_LABELS``.
        bad_ch_names : list | tuple
            Channel names. Will be the values corresponding to the ``kind``
            dictionary key.

        Returns
        -------
        None
        """
        logger.debug(f"NEW BAD CHANNELS {bad_ch_names}")
        if isinstance(bad_ch_names, xr.DataArray):
            bad_ch_names = bad_ch_names.values
        if kind in self:
            self[kind] = list(np.unique(np.concatenate((self[kind], bad_ch_names))))
        else:
            self[kind] = bad_ch_names


    def get_flagged(self):
        """Return a list of channels flagged by the lossless pipeline."""
        flagged_chs = list(self.values())
        if not flagged_chs:
            return []
        flagged_chs = np.unique(np.concatenate(flagged_chs))  # drop duplicates
        return flagged_chs.tolist()

    def rereference(self, inst, **kwargs):
        """Re-reference the Raw object attached to the LosslessPipeline.

        Parameters
        ----------
        inst : mne.io.Raw
            An instance of :class:`~mne.io.Raw` that contains EEG channels.
        kwargs : dict
            dictionary of valid keyword arguments for the
            :meth:`~mne.io.Raw.set_eeg_reference` method.
        """
        # Concatenate and remove duplicates
        bad_chs = list(
            set(self.ll.find_outlier_chs(inst, picks="eeg") +
                self.get_flagged() +
                inst.info["bads"])
        )
        ref_chans = [ch for ch in inst.copy().pick("eeg").ch_names
                     if ch not in bad_chs]
        inst.set_eeg_reference(ref_channels=ref_chans, **kwargs)

    def save_tsv(self, fname):
        """Save flagged channel annotations to a text file.

        Parameters
        ----------
        fname : str
            Filename that the annotations will be saved to.
        """
        labels = []
        ch_names = []
        for key in self:
            labels.extend([key] * len(self[key]))
            ch_names.extend(self[key])
        pd.DataFrame({"labels": labels, "ch_names": ch_names}).to_csv(
            fname, index=False, sep="\t"
        )

    def load_tsv(self, fname):
        """Load serialized channel annotations.

        Parameters
        ----------
        fname : str
            Filename of the tsv file with the annotation information to be
            loaded.
        """
        out_df = pd.read_csv(fname, sep="\t")
        for label, grp_df in out_df.groupby("labels"):
            self[label] = grp_df.ch_names.values


class FlaggedEpochs(_Flagged):
    """Object for handling flagged Epochs in an instance of mne.Epochs.

    Methods
    -------
    add_flag_cat:
        Append a list of indices (corresponding to Epochs in an instance of
        :class:`mne.Epochs`) to the ``'manual'`` dictionary key.

    load_from_raw:
        Add any pylossless :class:`mne.Annotations` in a loaded :class:`mne.io.Raw`
        file to the FlaggedEpochs class.

    Notes
    -----
    This class inherits from :class:`dict`, and can use any valid attributes
    and methods for python dictionaries.
    """

    def __init__(self, *args, **kwargs):
        """Initialize class.

        Parameters
        ----------
        ll : LosslessPipeline
            Instance of the lossless pipeline.
        args : list | tuple
            positional arguments accepted by `dict` class
        kwargs : dict
            keyword arguments accepted by python's dictionary class.
        """
        super().__init__(EPOCH_LABELS, "epoch", *args, **kwargs)

    def add_flag_cat(self, kind, bad_epoch_inds, epochs):
        """Add information on time periods flagged by pyLossless.

        Parameters
        ----------
        kind : str
            Should be one of the values in ``EPOCH_LABELS``.
        bad_epochs_inds : list | tuple
            Indices for the epochs in an :class:`mne.Epochs` object. Will be
            the values for the ``kind`` dictionary key.
        raw : mne.io.Raw
            The mne Raw object that is being assesssed by the LosslessPipeline
        epochs : mne.Epochs
            The :class:`mne.Epochs` object created from the Raw object that is
            being assessed by the LosslessPipeline.
        """
        if kind in self:
            self[kind] = list(np.unique(np.concatenate((self[kind], bad_epoch_inds))))
        else:
            self[kind] = bad_epoch_inds
        self.ll.add_pylossless_annotations(bad_epoch_inds, kind, epochs)

    def load_from_raw(self, raw, events, config):
        """Load pylossless annotations from raw object."""
        sfreq = raw.info["sfreq"]
        tmax = config["epoching"]["epochs_args"]["tmax"]
        tmin = config["epoching"]["epochs_args"]["tmin"]
        starts = events[:, 0] / sfreq - tmin
        stops = events[:, 0] / sfreq + tmax
        for annot in raw.annotations:
            if annot["description"].upper().startswith("BAD_LL_"):
                onset = annot["onset"]
                offset = annot["onset"] + annot["duration"]
                mask = (
                    (starts >= onset) & (starts < offset)
                    | (stops > onset) & (stops <= offset)
                    | (onset <= starts) & (offset >= stops)
                )
                inds = np.where(mask)[0]
                desc = annot["description"].lower().replace("bad_ll_", "")
                if desc not in self:
                    self[desc] = np.array([])
                self[desc] = np.concatenate((self[desc], inds))


class FlaggedICs(pd.DataFrame):
    """Object for handling IC classification in an mne ICA object.

    Attributes
    ----------
    fname : pathlib.Path
        Filepath to the ``derivatives/pylosssless`` folder in the ``bids_root``
        directory.
    ica : mne.preprocessing.ICA
        An ICA object created by mne.preprocessing.ICA.
    data_frame : pandas.DataFrame
        A pandas DataFrame that contains the dictionary returned by
        :func:`mne_icalabel.label_components`.

    Methods
    -------
    label_components :
        Labels the independent components in an instance of
        :class:`mne.preprocessing.ICA` using
        :func:`mne_icalabel.label_components`. Assigns the returned dictionary
        to the ``data_frame`` attribute.
    save_tsv :
        Save the ic_labels returned by
        :func:`mne_icalabel.annotation.write_components_tsv`
        to the `derivatives/pylossless` folder in the `bids_root` directory.
    load_tsv :
        Load flagged independent components that were previously saved to a
        tsv file.
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

    def label_components(self, epochs, ica):
        """Classify components using mne_icalabel.

        Parameters
        ----------
        epochs : mne.Epochs
            instance of `mne.Epochs` to be passed into
            :func:`mne_icalabel.label_components`.
        ica : mne.ICA
            instance of `mne.ICA` to be passed into
            :func:`mne_icalabel.label_components`.
        method : str (default "iclabel")
            The proposed method for labeling components, to be passed into
            :func:`mne_icalabel.label_components`. Must be one of: `'iclabel'`.
        """
        mne_icalabel.label_components(epochs, ica, method="iclabel")
        self.__init__(_icalabel_to_data_frame(ica))

    def save_tsv(self, fname):
        """Save IC labels.

        Parameters
        ----------
        fname : str | pathlib.Path
            The output filename.
        """
        self.fname = fname
        self.to_csv(fname, sep="\t", index=False, na_rep="n/a")

    # TODO: Add parameters.
    def load_tsv(self, fname, data_frame=None):
        """Load flagged ICs from file."""
        self.fname = fname
        if data_frame is None:
            data_frame = pd.read_csv(fname, sep="\t")
        self.__init__(data_frame)
