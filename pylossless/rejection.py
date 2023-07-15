# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#
# License: MIT

"""Classe to implement a policy on how to apply flags to reject artifacts."""

import numpy as np

from .config import Config


class RejectionPolicy:
    """Class used to implement a rejection policy for a pipeline output."""

    def __init__(self, config_fname=None):
        """Initialize class.

        Parameters
        ----------
        config_fname : pathlib.Path
            path to config file specifying the parameters to be used
            in for the rejection policy.
        """
        self.ch_flags_to_reject = ['ch_sd', 'low_r', 'bridged']
        self.epoch_flags_to_reject = []
        self.ic_flags_to_reject = ['muscle', 'heart', 'ch_noise', 'eye']

        self.ic_rejection_threshold = 0.3

        # Can be:
        #   "": Do nothing aside from adding them to raw.info["bads"]
        #   "drop": Drop bad channels.
        #   "interpolate": Interpolate bad channels.
        self.clean_ch_mode = ""

        self.remove_bad_ic = True

        self.interpolate_bads_kwargs = {}

        if config_fname is not None:
            config = Config().read(config_fname)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def apply(self, pipeline):
        """Apply the rejection policy to the output of the lossless run.

        Parameters
        ----------
        pipeline : LosslessPipeline
            An instance of LosslessPipeline with after the pipeline has
            been ran.

        Returns
        -------
        mne.Raw
            An mne.Raw instance with the appropriate channels and ICs
            rejected and the bad segments marked as bad annotations.
        """
        # Get the raw object
        raw = pipeline.raw.copy()

        # Add channels to be rejected as bads
        for key in self.ch_flags_to_reject:
            if key in pipeline.flags["ch"]:
                raw.info["bads"] += pipeline.flags["ch"][key].tolist()

        # Clean the channels
        if self.clean_ch_mode == "drop":
            raw.drop_channels(raw.info["bads"])
        elif self.clean_ch_mode == "interpolate":
            raw.interpolate_bads(**self.interpolate_bads_kwargs)

        # Clean the epochs
        # TODO: Not sure where we landed on having these prefixed as bad_
        #       or not by the pipeline. If not prefixed, this would be the
        #       step that add select types of flags as bad_ annotations.

        # Clean the ics
        ic_labels = pipeline.flags['ic'].data_frame
        mask = np.array([False]*len(ic_labels['confidence']))
        for label in self.ic_flags_to_reject:
            mask |= ic_labels['ic_type'] == label
        mask &= ic_labels['confidence'] > self.ic_rejection_threshold

        flagged_ics = ic_labels.loc[mask]
        if not flagged_ics.empty:
            flagged_ics = flagged_ics.index.tolist()
            pipeline.ica2.exclude.extend(flagged_ics)
            pipeline.ica2.apply(raw)

        return raw
