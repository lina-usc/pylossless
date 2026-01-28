# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#
# License: MIT

import numpy as np
from importlib.metadata import version
import warnings

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
        ``["noisy", "uncorrelated", "bridged"]``, meaning that any channels that
        have been flagged ``"noisy"``, ``"uncorrelated"``, or ``"bridged"`` will
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
        # Preprocessing policy attributes
        self.apply_preprocessing = True
        self.preprocessing_operations_to_skip = []
        self.operation_param_overrides = {}

        if ch_flags_to_reject == "all":
            ch_flags_to_reject = ["noisy", "uncorrelated", "bridged"]
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
        """Return a summary of the RejectionPolicy object."""
        return (
            f"RejectionPolicy: |\n"
            f"  config_fname: {self['config_fname']}\n"
            f"  ch_flags_to_reject: {self['ch_flags_to_reject']}\n"
            f"  ic_flags_to_reject: {self['ic_flags_to_reject']}\n"
            f"  ic_rejection_threshold: {self['ic_rejection_threshold']}\n"
            f"  ch_cleaning_mode: {self['ch_cleaning_mode']}\n"
            f"  remove_flagged_ics: {self['remove_flagged_ics']}\n"
        )

    def apply(self, pipeline, return_ica=False, version_mismatch="raise"):
        """Return a cleaned new raw object based on the rejection policy.

        This method replays all operations from the pipeline's operation log
        in the exact order they were executed, ensuring reproducibility.

        Parameters
        ----------
        pipeline : LosslessPipeline
            An instance of LosslessPipeline after the pipeline has been run.
        return_ica : bool
            If ``True``, returns the :class:`~mne.preprocessing.ica` object
            used to clean the :class:`~mne.io.Raw` object. Defaults to False.
        version_mismatch : str
            How to handle version mismatches. One of 'raise', 'warning', or
            'ignore'.

        Returns
        -------
        mne.io.Raw
            An :class:`~mne.io.Raw` instance with the appropriate channels
            and ICs added to mne bads, interpolated, or dropped.
        """
        from mne.utils import logger

        if pipeline.config["version"] != version("pylossless"):
            error_message = (
                "The output of the pipeline was saved with pylossless version "
                f"{pipeline.config['version']} and you are currently using "
                f"version {version('pylossless')}. Behavior is undefined."
            )
            if version_mismatch == "raise":
                raise RuntimeError(error_message)
            elif version_mismatch == "warning":
                warnings.warn(error_message, RuntimeWarning)
            elif version_mismatch != "ignore":
                raise ValueError(
                    "version_mismatch can take values 'raise', 'warning', or "
                    f"'ignore'. Received {version_mismatch}."
                )

        # Check if we have operations log (new lossless format)
        if hasattr(pipeline, 'operations_log') and len(pipeline.operations_log) > 0:
            logger.info("LOSSLESS: Applying rejection policy by replaying operations...")
            raw = self._apply_with_replay(pipeline)
        else:
            # Fallback to old method for backwards compatibility
            logger.info("LOSSLESS: Using legacy apply method (no operations log)")
            raw = self._apply_legacy(pipeline)

        if return_ica:
            return raw, pipeline.ica2
        return raw

    def _apply_legacy(self, pipeline):
        """Legacy apply method for backwards compatibility."""
        # Get the raw object
        raw = pipeline.raw.copy()

        # Add channels to be rejected as bads
        for key in self["ch_flags_to_reject"]:
            if key in pipeline.flags["ch"]:
                raw.info["bads"] += pipeline.flags["ch"][key].tolist()

        # Clean the channels
        if self["ch_cleaning_mode"] == "drop":
            raw.drop_channels(raw.info["bads"])
        elif self["ch_cleaning_mode"] == "interpolate":
            raw.interpolate_bads(**self["interpolate_bads_kwargs"])

        # Clean the ics
        ic_labels = pipeline.flags["ic"]
        mask = np.array([False] * len(ic_labels["confidence"]))
        for label in self["ic_flags_to_reject"]:
            mask |= ic_labels["ic_type"] == label
        mask &= ic_labels["confidence"] > self["ic_rejection_threshold"]

        flagged_ics = ic_labels.loc[mask]
        if not flagged_ics.empty:
            flagged_ics = flagged_ics.index.tolist()
            pipeline.ica2.exclude.extend(flagged_ics)
            pipeline.ica2.apply(raw)

        return raw

    def _apply_with_replay(self, pipeline):
        """Apply rejection policy by replaying operations from log."""
        from mne.utils import logger
        from mne.preprocessing import ICA

        # Start with original unmodified data
        raw = pipeline.raw_original.copy()

        # Ensure data is preloaded for operations
        if not raw.preload:
            raw.load_data()

        # ICA model (will be fitted during replay if needed)
        ica_model = None

        # Track what was done
        n_preprocessing_applied = 0
        n_preprocessing_skipped = 0

        # Replay all operations in order
        for operation in pipeline.operations_log:
            op_id = operation["operation_id"]
            op_type = operation["operation_type"]
            op_name = operation["operation_name"]
            params = operation["parameters"]

            # PREPROCESSING OPERATIONS
            if op_type == "preprocessing":
                if not self.apply_preprocessing:
                    logger.info(
                        f"  Op {op_id}: Skipping {op_name} "
                        "(preprocessing disabled)"
                    )
                    n_preprocessing_skipped += 1
                    continue

                if op_name in self.preprocessing_operations_to_skip:
                    logger.info(
                        f"  Op {op_id}: Skipping {op_name} (in skip list)"
                    )
                    n_preprocessing_skipped += 1
                    continue

                # Get final parameters (potentially overridden)
                final_params = self._get_final_params(op_name, params)

                logger.info(f"  Op {op_id}: Applying {op_name}...")

                if op_name == "filter":
                    raw.filter(**final_params)
                    n_preprocessing_applied += 1

                elif op_name == "notch_filter":
                    raw.notch_filter(**final_params)
                    n_preprocessing_applied += 1

                elif op_name == "set_eeg_reference":
                    # Update exclude list based on rejection policy
                    if "exclude" in final_params:
                        excluded_chs = self._get_channels_to_exclude_at_operation(
                            pipeline, op_id
                        )
                        final_params["exclude"] = excluded_chs
                        logger.info(
                            f"         Excluding {len(excluded_chs)} "
                            "channels from reference"
                        )

                    raw.set_eeg_reference(**final_params)
                    n_preprocessing_applied += 1

                elif op_name == "resample":
                    raw.resample(**final_params)
                    n_preprocessing_applied += 1

                else:
                    logger.warning(
                        f"  Op {op_id}: Unknown preprocessing "
                        f"operation: {op_name}"
                    )

            # ICA FIT OPERATIONS
            elif op_type == "ica_fit":
                if self["remove_flagged_ics"]:
                    logger.info(
                        f"  Op {op_id}: Fitting ICA on current data state..."
                    )
                    ica_model = self._fit_ica(raw, params)
                else:
                    logger.info(
                        f"  Op {op_id}: Skipping ICA (removal disabled)"
                    )

            # Other operation types are just noted
            elif op_type in ["artifact_flag", "ica_label"]:
                logger.debug(f"  Op {op_id}: Noted {op_name}")
                pass

        logger.info(
            f"  Preprocessing: {n_preprocessing_applied} applied, "
            f"{n_preprocessing_skipped} skipped"
        )

        # Apply final artifact rejections
        logger.info("LOSSLESS: Applying final artifact rejections...")

        # Apply channel rejections
        channels_to_reject = self._get_channels_to_reject(pipeline)
        if channels_to_reject:
            if self["ch_cleaning_mode"] is None:
                logger.info(
                    f"  Flagging {len(channels_to_reject)} channels as bad"
                )
                raw.info["bads"].extend(channels_to_reject)
            elif self["ch_cleaning_mode"] == "interpolate":
                logger.info(
                    f"  Interpolating {len(channels_to_reject)} channels"
                )
                raw.info["bads"].extend(channels_to_reject)
                raw.interpolate_bads(**self["interpolate_bads_kwargs"])
            elif self["ch_cleaning_mode"] == "drop":
                logger.info(
                    f"  Dropping {len(channels_to_reject)} channels"
                )
                raw.info["bads"].extend(channels_to_reject)
                raw.drop_channels(raw.info["bads"])

        # Apply ICA component removal
        if self["remove_flagged_ics"] and ica_model is not None:
            components_to_remove = self._get_ica_components_to_remove(pipeline)
            if components_to_remove:
                logger.info(
                    f"  Removing {len(components_to_remove)} ICA components"
                )
                ica_model.apply(raw, exclude=components_to_remove)

        logger.info("LOSSLESS: ✓ Rejection policy applied successfully")
        return raw

    def _get_final_params(self, op_name, original_params):
        """Get final parameters for an operation with potential overrides.

        Parameters
        ----------
        op_name : str
            Operation name
        original_params : dict
            Original parameters from operation log

        Returns
        -------
        final_params : dict
            Final parameters (potentially overridden)
        """
        from mne.utils import logger

        if op_name in self.operation_param_overrides:
            # Override parameters
            overridden = original_params.copy()
            overridden.update(self.operation_param_overrides[op_name])
            logger.debug(f"    Using overridden parameters for {op_name}")
            return overridden
        return original_params

    def _get_channels_to_exclude_at_operation(self, pipeline, current_op_id):
        """Get channels to exclude for re-referencing at a given operation.

        This is critical for handling operation dependencies: re-referencing
        operations need to know which channels were flagged in previous
        operations.

        Parameters
        ----------
        pipeline : LosslessPipeline
            Pipeline object with operation log
        current_op_id : int
            Current operation ID

        Returns
        -------
        excluded_channels : list
            List of channel names to exclude
        """
        excluded_channels = []

        # Look at all operations before current one
        for op in pipeline.operations_log:
            if op["operation_id"] >= current_op_id:
                break

            if op["operation_type"] == "artifact_flag":
                # Check for flagged channels
                if "noisy_channels" in op["flags"]:
                    flagged = op["flags"]["noisy_channels"]
                    for ch in flagged:
                        if "noisy" in self["ch_flags_to_reject"]:
                            if ch not in excluded_channels:
                                excluded_channels.append(ch)

                # Check for bridged channels
                if "bridged_channels" in op["flags"]:
                    flagged = op["flags"]["bridged_channels"]
                    for ch in flagged:
                        if "bridged" in self["ch_flags_to_reject"]:
                            if ch not in excluded_channels:
                                excluded_channels.append(ch)

                # Check for uncorrelated channels
                if "uncorrelated_channels" in op["flags"]:
                    flagged = op["flags"]["uncorrelated_channels"]
                    for ch in flagged:
                        if "uncorrelated" in self["ch_flags_to_reject"]:
                            if ch not in excluded_channels:
                                excluded_channels.append(ch)

        return excluded_channels

    def _get_channels_to_reject(self, pipeline):
        """Get all channels to reject based on policy.

        Parameters
        ----------
        pipeline : LosslessPipeline
            Pipeline object with operation log

        Returns
        -------
        channels_to_reject : list
            List of channel names to reject
        """
        channels_to_reject = set()

        for op in pipeline.operations_log:
            if op["operation_type"] == "artifact_flag":
                # Check for noisy channels
                if "noisy_channels" in op["flags"]:
                    if "noisy" in self["ch_flags_to_reject"]:
                        channels_to_reject.update(op["flags"]["noisy_channels"])

                # Check for bridged channels
                if "bridged_channels" in op["flags"]:
                    if "bridged" in self["ch_flags_to_reject"]:
                        channels_to_reject.update(op["flags"]["bridged_channels"])

                # Check for uncorrelated channels
                if "uncorrelated_channels" in op["flags"]:
                    if "uncorrelated" in self["ch_flags_to_reject"]:
                        channels_to_reject.update(
                            op["flags"]["uncorrelated_channels"]
                        )

        return list(channels_to_reject)

    def _get_ica_components_to_remove(self, pipeline):
        """Get ICA components to remove based on policy.

        Parameters
        ----------
        pipeline : LosslessPipeline
            Pipeline object with operation log

        Returns
        -------
        components_to_remove : list
            List of component indices to remove
        """
        components_to_remove = []

        for op in pipeline.operations_log:
            if op["operation_type"] == "ica_label":
                ic_labels = op["flags"].get("ic_labels", {})
                for comp_id, label_info in ic_labels.items():
                    label = label_info.get("ic_type")
                    confidence = label_info.get("confidence", 0)

                    if (
                        label in self["ic_flags_to_reject"]
                        and confidence >= self["ic_rejection_threshold"]
                    ):
                        components_to_remove.append(int(comp_id))

        return components_to_remove

    def _fit_ica(self, raw, params):
        """Fit ICA model on data.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data to fit ICA on
        params : dict
            ICA parameters

        Returns
        -------
        ica : mne.preprocessing.ICA
            Fitted ICA model
        """
        from mne.preprocessing import ICA
        from mne.utils import logger

        ica = ICA(
            n_components=params.get("n_components", 25),
            method=params.get("method", "fastica"),
            max_iter=params.get("max_iter", 200),
            random_state=params.get("random_state", 42),
        )

        logger.debug(f"    Fitting ICA with {ica.n_components} components...")
        ica.fit(raw)

        return ica
