"""

Run pyLossless on a BIDS dataset.
=================================

In this notebook, we will run the pyLossless pipeline on a publicly available dataset
that is already in BIDS format.
"""

# %%
# Imports
# -------
from pathlib import Path
import shutil
import pylossless as ll

# %%
# Get the data
# ------------
raw, config, bids_path = ll.datasets.load_openneuro_bids()

# %%
# Prep the Raw object
# -------------------
#
# This data has EOG channels that are not labeled as such. We will manually set the
# channel types to be "eog" for these channels (i.e. "EXG1"). We will also crop the
# data to 60 seconds for speed, and load the data in memory, which is required for
# running the pipeline.
raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch.startswith("EX")})
raw.load_data().crop(0, 60)

# %%
# Initialize the pipeline
# -----------------------
#
# The :class:`~pylossless.LosslessPipeline` instance is the main object that will
# run the pipeline. It takes a file path to a :class:`~pylossless.config.Config` object
# as input. :func:`~pylossless.datasets.load_openneuro_bids` returned a
# :class:`~pylossless.config.Config` object, so we will save it to disk and pass the
# file path to the :class:`~pylossless.LosslessPipeline` constructor.
config_path = Path("lossless_config.yaml")
config["filtering"]["notch_filter_args"]["freqs"] = [60]
config.save(config_path)
pipeline = ll.LosslessPipeline(config_path)

# %%
# Run the pipeline
# ----------------
#
# The :class:`~pylossless.LosslessPipeline` object has a
# :meth:`~pylossless.LosslessPipeline.run_with_raw` method that takes a
# :class:`~mne.io.Raw` object as input.
# We will use the :class:`~mne.io.Raw` object that was returned by
# :func:`~pylossless.datasets.load_openneuro_bids` with the pipeline.
pipeline.run_with_raw(raw)

# %%
# View the results
# ----------------
#
# The :class:`~pylossless.LosslessPipeline` object stores flagged channels and ICs in
# the :attr:`~pylossless.LosslessPipeline.flags` attribute:
print(f"flagged channels: {pipeline.flags['ch']}")
print(f"flagged ICs: {pipeline.flags['ic'].data_frame}")

# %%
# Get the cleaned data
# --------------------
#
# The :class:`~pylossless.LosslessPipeline` by default does not modify the
# :class:`~mne.io.Raw` object that is passed to it, so none of the flagged channels
# or ICs are removed from the :class:`~mne.io.Raw` object yet. To get the cleaned
# :class:`~mne.io.Raw` object, we need to call the
# :meth:`~pylossless.LosslessPipeline.make_cleaned_raw` method. This method takes a
# :class:`~pylossless.RejectionPolicy` as input, which specifies how to apply the flags
# to generate a new :class:`~mne.io.Raw` object.
rejection_policy = ll.RejectionPolicy()
rejection_policy["ch_cleaning_mode"] = "interpolate"
rejection_policy

# %%
# We set the channel cleaning mode to ``"interpolate"``, which means that the flagged
# channels will be interpolated. Similar to the :class:`~pylossless.config.Config`,
# We need to save this :class:`~pylossless.RejectionPolicy` to disk, and pass the file
# to return a new cleaned :class:`~mne.io.Raw` object:
rejection_policy_path = Path("rejection_policy.yaml")
rejection_policy.save(rejection_policy_path)
cleaned_raw = pipeline.make_cleaned_raw(rejection_policy_path)
cleaned_raw.plot()

# %%
# Save the PyLossless Derivative
# ------------------------------
#
# Let's save our pipeline output to disk. We need to use our
# :class:`~mne_bids.BIDSPath` object to set up a derivatives path to save the
# pipeline output to:
derivatives_path = pipeline.get_derivative_path(bids_path)
derivatives_path

# %%
pipeline.save(derivatives_path)

# %%
# Clean up
# --------
#
shutil.rmtree(bids_path.root)
config_path.unlink()
rejection_policy_path.unlink()
