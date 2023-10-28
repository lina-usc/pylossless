# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#
# License: MIT

from .config import ConfigMixin


class RejectionPolicy(ConfigMixin):
    """Class used to implement a rejection policy for a pipeline output.

    Parameters
    ----------
    config_fname : pathlib.Path
        path to config file specifying the parameters to be used
        in for the rejection policy.
    ch_flags_to_reject : list of str
        List of channel flags to apply. "all" is the same as
        ``["ch_sd", "low_r", "bridge"]``, meaning that any channels that
        have been flagged ``"ch_sd"``, ``"low_r"``, or ``"bridge"`` will
        be applied. Defaults to "all".
    ic_flags_to_reject : list of str
        List of IC flags to apply. "all" is the same as
        ``["muscle", "heart", "eye", "channel noise", "line noise"]``, meaning that
        any ICs that have been flagged ``"muscle"``, ``"ecg"``, ``"eog"``
        ``"channel_noise"``, or ``"line_noise"`` will be applied, if their label
        confidence is greater than ``ic_rejection_threshold``.
        Note this list does NOT include ``"brain"`` and ``"other"``. Defaults to
        "all".
    ic_rejection_threshold : float
        threshold (between 0 and 1), representing the minimum confidence
        percentage in a given label (i.e., .30 is 30% confidence). For any labels
        passed to ``ic_flags_to_reject`` (i.e. ``"ecg"``, ``"eog"``, etc.), ICs with
        that label and a confidence percentage greater than this threshold will
        be rejected. Defaults to ``0.3``.
    ch_cleaning_mode : str
        Must be one of ``None``, ``"drop"``, or ``"interpolate"``. ``None`` adds the
        channels to ``raw.info["bads"]``. ``"drop"`` drops the channels from the
        :class:`~mne.io.Raw` object. ``"interpolate"`` interpolates the bad channels.
        Defaults to ``None``.
    interpolate_bads_kwargs : None | dict
        If ``ch_cleaning_mode`` is ``"interpolate"``, these keyword arguments
        will be passed to ``raw.interpolate_bads(**interpolate_bads_kwargs)``.
        Must be a dictionary of valid keyword arguments for
        :meth:`~mne.io.Raw.interpolate_bads`. Defaults to ``None``, which means no
        keyword arguments will be passed.
    remove_flagged_ics : bool
        If ``True``, subtracts the signal accounted for by the flagged ICs
        from the ``raw`` object, via :meth:`~mne.preprocessing.ICA.apply`.
        If ``False``, does nothing. Defaults to ``True``.

    """

    def __init__(
        self,
        *,
        config_fname=None,
        ch_flags_to_reject="all",
        ic_flags_to_reject="all",
        ic_rejection_threshold=0.3,
        ch_cleaning_mode=None,
        interpolate_bads_kwargs=None,
        remove_flagged_ics=True,
    ):
        if ch_flags_to_reject == "all":
            ch_flags_to_reject = ["ch_sd", "low_r", "bridge"]
        if ic_flags_to_reject == "all":
            ic_flags = ["muscle", "ecg", "eog", "channel_noise", "line_noise"]
            ic_flags_to_reject = ic_flags

        if interpolate_bads_kwargs is None:
            interpolate_bads_kwargs = {}

        if config_fname is not None:
            config = ConfigMixin().read(config_fname)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        super().__init__(
            config_fname=config_fname,
            ch_flags_to_reject=ch_flags_to_reject,
            ic_flags_to_reject=ic_flags_to_reject,
            ic_rejection_threshold=ic_rejection_threshold,
            ch_cleaning_mode=ch_cleaning_mode,
            interpolate_bads_kwargs=interpolate_bads_kwargs,
            remove_flagged_ics=remove_flagged_ics,
        )

    def __repr__(self):
        """Return a summary of the Calibration object."""
        return (
            f"RejectionPolicy: |\n"
            f"  config_fname: {self['config_fname']}\n"
            f"  ch_flags_to_reject: {self['ch_flags_to_reject']}\n"
            f"  ic_flags_to_reject: {self['ic_flags_to_reject']}\n"
            f"  ic_rejection_threshold: {self['ic_rejection_threshold']}\n"
            f"  ch_cleaning_mode: {self['ch_cleaning_mode']}\n"
            f"  remove_flagged_ics: {self['remove_flagged_ics']}\n"
        )