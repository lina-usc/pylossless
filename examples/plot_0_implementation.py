r"""

PyLossless Algorithms
=====================

This tutorial explains the calculations that PyLossless performs at each step of the
pipeline. We will use example EEG data to demonstrate the
calculations.

.. note::
    You can open this notebook in
    `Google Colab <https://colab.research.google.com/drive/1ecyNo10oFgpbVNuD7Ztgr2fs8XkpfOYo?usp=sharing>`_!

.. _notation:

Notation 
--------
Before we begin, we define some notation that will be used throughout the text:

- We start with a 3D matrix of EEG data,
  :math:`X \in \mathbb{R}^{S_\mathcal{G} \times E_\mathcal{G} \times T}`,
  where :math:`S_\mathcal{G}` and :math:`E_\mathcal{G}` are the sets of good sensors and
  epochs, respectively, and :math:`T`, is the number of samples(i.e. time-points).

- ``s``, ``e``, and ``t`` are sensor, epochs, and samples, respectively.

- We use superscripts to denote operations across a dimension, and we use subscripts to
  denote indexing a dimension.

- We refer to a single sensor :math:`i` as
  :math:`X\big|_{s=i} \in \mathbb{R}^{E_\mathcal{G} \times T}`,
  with :math:`i \in S_\mathcal{G}`.

- We refer to a single epoch :math:`j` as
  :math:`X\big|_{e=j} \in \mathbb{R}^{S_\mathcal{G} \times T}`,
  with :math:`j \in E_\mathcal{G}`.

- We denote sensor-specific thresholds for rejecting epochs as
  :math:`\tau^e_i \in \mathbb{R}^{S_\mathcal{G}}`

- We denote epoch-specific thresholds for rejecting sensors as
  :math:`\tau^s_j \in \mathbb{R}^{E_\mathcal{G}}`

- We denote *quantiles* as :math:`Q\#^{dim}`: i.e. :math:`Q75^s` is the 75th *quantile*
  along the sensor dimension. The function :math:`Q75^s(X)` computes the 75th quantile
  along the :math:`s` dimension of matrix :math:`X`, resulting in a matrix noted
  :math:`X^{Q75^s} \in \mathbb{R}^{E \times T}`.

Throughout the text, we use capital letters for matrices and lowercase letters for
scalars. For example, the data point for sensor :math:`i`, epoch :math:`j`, and
time :math:`k` is denoted as :math:`X\big|_{s=i; e=j; t=k} = x_{ijk} \in \mathbb{R}`,
and :math:`X=\{x_{ijk}\}`.
"""

# %%
# Imports and data loading
# ------------------------
from pathlib import Path

import numpy as np

import mne
from mne.datasets import sample

import pylossless as ll

# Load example mne data
raw = ll.datasets.load_simulated_raw()

# %%
# Load a PyLossless configuration file
# ------------------------------------
# Let's load a PyLossless configuration file. This file contains the parameters that
# will be used for each step of the pipeline. For example, the ``noisy_channels``
# section contains the parameters for the :ref:`noisy_sensors` step. We can modify
# these parameters to change the behavior of the pipeline. For example, we can change
# the percent of epochs that a sensor must be noisy for it to be flagged via the
# ``flag_crit`` parameter.
config = ll.config.Config()
config.load_default()
config["noisy_channels"]["outliers_kwargs"]["lower"] = 0.25  # lower quantile
config["noisy_channels"]["outliers_kwargs"]["upper"] = 0.75  # upper quantile
config["noisy_channels"]["flag_crit"] = 0.30  # percent of epochs that a sensor must be noisy
config.save("test_config.yaml")

# %%
# Create a pipeline instance
# --------------------------
pipeline = ll.LosslessPipeline("test_config.yaml")
pipeline.raw = raw
raw.plot()

# %%
# Input Data
# ----------
#
# First, we epoch the data to be used for subsequent steps.
# Let our 3D matrix below be defined as :math:`X \in \mathbb{R}^{S \times E \times T}`
# where :math:`X` is a matrix of real numbers and of dimension :math:`S` sensors
# :math:`\times$ E` epochs `\times T` times.
#
epochs = pipeline.get_epochs()

# %%
#
# Let's convert our epochs object into a named :class:`xarray.DataArray` object.
from pylossless.pipeline import epochs_to_xr

#
epochs_xr = epochs_to_xr(epochs, kind="ch")
epochs_xr  # 277 epochs, 50 sensors, 602 samples per epoch


# %%
# .. _robust_reference:
#
# Robust Average Reference
# ------------------------
#
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/robust_rereference.png
#    :align: center
#    :alt: Robust Average Reference graphic.
#
#    Robust Average Reference. The figure shows the steps for robust average referencing.
#    See the text below for descriptions of mathematical notation.
#
# Before the pipeline can begin, we must average reference the data. This is because
# the pipeline uses data distributions to identify noisy sensors, and For EEG data that
# uses an online reference to a single electrode, sensors that are further from the
# reference will have a higher voltage variance, and the pipeline will be biased to
# flag these sensors as noisy. The average reference, which subtracts the average
# signal across sensors from each individual sensor, will ensure an even playing field.
# Howeer, we dont want to include noisy sensors in the average reference signal. So we
# will identify noisy sensors and and leave them out of the average reference signal.

# %%
sample_std = epochs_xr.std("time")
q25_ch = sample_std.quantile(0.25, dim="ch")
q50_ch = sample_std.median(dim="ch")
q75_ch = sample_std.quantile(0.75, dim="ch")
ch_dist = sample_std - q50_ch  # center the data
ch_dist /= q75_ch - q25_ch  # shape (chans, epoch)

mean_ch_dist = ch_dist.mean(dim="epoch")  # shape (chans)

# find the median and 25 and 75 percentiles
# of the mean of the channel distributions
mdn = np.median(mean_ch_dist)
deviation = np.diff(np.quantile(mean_ch_dist, [0.25, 0.75]))

leave_out = mean_ch_dist.ch[mean_ch_dist > mdn + 6 * deviation].values.tolist()
leave_out

# %%
ref_chans = [ch for ch in epochs.pick("eeg").ch_names if ch not in leave_out]
pipeline.raw.set_eeg_reference(ref_channels=ref_chans)

# %%
#
# .. _noisy_sensors:
#
# Flag Noisy Sensors
# ------------------
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Flag_noisy_sensors.png
#    :align: center
#    :alt: Flag Noisy Sensors graphic.
#
#    Flag Noisy Sensors. The figure shows the steps for flagging noisy sensors. See the text below
#    for descriptions of mathematical notation.
#

# %%
# Since we applied a robust average reference to the raw data, we will need to re-epoch
# the data:
epochs = pipeline.get_epochs()
epochs_xr = epochs_to_xr(epochs, kind="ch")

# First we take standard deviation of
# :math:`X \in \mathbb{R}^{S \times E \times T}` across the samples dimension :math:`t`
# resulting in a 2D matrix :math:`X^{\sigma_{t}} \in \mathbb{R}^{S \times E}`
trim_ch_sd = epochs_xr.std("time")
trim_ch_sd

# %%
# a) Take the 50th and 75th quantile across dimension sensor of :math:`X^{\sigma_{t}}`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This operation results in two 1D vectors of size :math:`E`:
#
# .. math::
#    X^{{\sigma}_t{Q50^s}} = Q50^s(X^{\sigma_{t}}) \in \mathbb{R}^{E}
# .. math::
#    X^{{\sigma}_t{Q75^s}} = Q75^s(X^{\sigma_{t}}) \in \mathbb{R}^{E}

# %%
q50, q75 = trim_ch_sd.quantile([0.5, 0.75], dim="ch")
q50  # a 1D array of median standard deviation values across channels for each epoch

# %%
# b) Define an Upper Quantile Range as :math:`Q75 - Q50`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. math::
#    UQR^s = X^{{\sigma}_T{Q75}^s} - X^{{\sigma}_T{Q50}^s}
#
# This operation results in a 1D vector of size :math:`E`.
uqr = q75 - q50
uqr

# %%
# c) Identify outlier Indices :math:`(i, j)`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We multiply a constant :math:`k` by the :math:`UQR` to define a measure for the
# spread of the right tail of the distribution of :math:`X^{\sigma_{t}}` values and
# add it to the median of :math:`X^{\sigma_{t}}` to obtain epoch-specific standard
# deviation threshold for outliers:
#
# .. math::
#    \tau^s_j = X^{{\sigma}_T{Q50}^S} + UQR^s\times k
#
# That is, :math:`\tau^s_j` is the epoch-specific threshold for the epoch :math:`j`
k = 3
upper_threshold = q50 + q75 * k
upper_threshold  # epoch specific thresholds

# %%
# Now, we compare our 2D standard deviation matrix to the threshold vector of
# :math:`\tau^e_j`:
#
# .. math::
#    X^{\sigma_{t}} \big|_{e=j}  > \tau^s_j
#
# resulting in the indicator matrix :math:`C \in \{0, 1\}^{S \times E}=\{c_{ij}\}`:
#
# .. math::
#    c_{ij} =
#    \begin{cases}
#    0 & \text{if } x^{\sigma_{t}}_{ij} < \tau^s_j \\
#    1 & \text{if } x^{\sigma_{t}}_{ij} \geq \tau^s_j
#    \end{cases}
#
# Each element of this matrix indicates whether sensor :math:`i` is an outlier at an epoch
# :math:`j`.
outlier_mask = trim_ch_sd > upper_threshold
outlier_mask

# %%
# d) Identify noisy sensors part 1
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To identify outlier sensors, we average across the epoch dimension of our indicator
# matrix :math:`C` and obtain :math:`C^{\mu_e} \in \mathbb{R}^{S_\mathcal{G}}`, which
# is a vector of fractional numbers :math:`c^{\mu_e}_i` representing the percentage of
# epochs for which that sensor is an outlier.
percent_outliers = outlier_mask.astype(float).mean("epoch")
percent_outliers  # percent of epochs that sensor is an outlier

# %%
# e) Identify noisy sensors part 2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we define a threshold :math:`\tau^{p}` (:math:`p` for percentile;
# default, ``.20``) as a cutoff point for determining if a sensor should be marked
# artifactual. The sensor :math:`i` is flagged as noisy if
# :math:`c^{\mu_e}_i > \tau^{p}`. That is, if the sensor is an outlier for more than
# :math:`\tau^{p}` percent of the epochs, it is flagged as noisy.
p_threshold = config["noisy_channels"]["flag_crit"]  # 0.3, or 30%
noisy_chs = percent_outliers[percent_outliers > p_threshold].coords.to_index().values
noisy_chs

# %%
# f) Add the noisy sensors to the pipeline flags
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let's add the noisy sensors to the pipeline flags.
pipeline.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=noisy_chs)
pipeline.raw.info["bads"].extend(pipeline.flags["ch"]["noisy"].tolist())
pipeline.flags["ch"]

# %%
#
# .. _noisy_epochs:
#
# Flag Noisy Epochs
# -----------------
#
# This step closely resembles the :ref:`noisy_sensors` step. For the sake of brevity
# we will be more concise in the documentation.

# %%
#
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Flag_noisy_epochs.png
#    :align: center
#
#    Flag Noisy Epochs. The figure shows the steps for flagging noisy epochs. See the text below
#    for descriptions of mathematical notation.
#

# %%
# a) Take standard deviation across the samples dimension :math:`t`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Take a moment below to notice that the sensors flagged in the prior setp are not in
# ``epochs_xr`` below:
epochs = pipeline.get_epochs()
# Let's make our epochs array into a named Array
epochs_xr = epochs_to_xr(epochs, kind="ch")
trim_ch_sd = epochs_xr.std("time")
trim_ch_sd.coords["ch"]

# %%
# b) Compute 50th and 75th quantile across epochs and the UQR
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Like before, We Take the median and 70th quantile, but now we operate across epochs,
# resulting in two 1D vector's of size ``n_good_sensors`` :math:`S_\mathcal{G}`
#
# .. math::
#    X^{{\sigma}_t{Q50^e}} = Q50^e(X^{\sigma_{t}}) \in \mathbb{R}^{S_\mathcal{G}}
# .. math::
#    X^{{\sigma}_t{Q75^e}} = Q75^e(X^{\sigma_{t}}) \in \mathbb{R}^{S_\mathcal{G}}
# .. math::
#    UQR^e = (X^{{\sigma}_T{Q75}^e} - X^{{\sigma}_T{Q50}^e})


# %%
# c) Define sensor-specific thresholds for rejecting epochs :math:`\tau^e_i`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Our sensor-specifc threshold for rejecting epochs is defined by:
#
# .. math::
#    \tau^e_i = X^{{\sigma}_T{Q50}^e} + UQR^e\times k
q50, q75 = trim_ch_sd.quantile([0.5, 0.75], dim="epoch")
uqr_epoch = q75 - q50
uqr_epoch

# %%
k = 8
upper_threshold = q50 + uqr_epoch * k
upper_threshold

# %%
# d) Identify Outlier indices
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The indicator matrix is defined by:
#
# .. math::
#    c_{ij} =
#    \begin{cases}
#    0 & \text{if } x^{\sigma_{t}}_{ij} < \tau^e_i \\
#    1 & \text{if } x^{\sigma_{t}}_{ij} \geq \tau^e_i
#    \end{cases}
#
#
# To identify outlier **epochs**, we average across the **sensor** dimension of our
# indicator matrix :math:`C` and obtain
# :math:`C^{\mu_s} \in \mathbb{R}^{E_\mathcal{G}}`, which is a vector of numbers
# :math:`c^{\mu_s}_j` representing the percentage of **sensors** for which that epoch
# is an outlier.
outlier_mask = trim_ch_sd > upper_threshold
outlier_mask

# %%
percent_outliers = outlier_mask.astype(float).mean("ch")
percent_outliers

# %%
# e) Identify noisy epochs
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we define a fractional threshold :math:`\tau^{p}` as a cutoff point for
# determining if a epoch should be marked artifactual. The epoch :math:`j` is flagged
# as noisy if :math:`c^{\mu_s}_j > \tau^{p}`.
bad_epochs = percent_outliers[percent_outliers > p_threshold].coords.to_index().values
bad_epochs

# %%
# f) Add the noisy epochs to the pipeline flags
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let's add the outlier epochs to our flags
# These will be added directly as :class:`mne.Annotations` to the raw data.
pipeline.flags["epoch"].add_flag_cat(
    kind="noisy", bad_epoch_inds=bad_epochs, epochs=epochs
)
pipeline.raw.annotations.description

# %%
pipeline.raw.plot()

# %%
# Filtering
# ---------
#
# After flagging noisy sensors and epochs, we filter the data. By default,
# The pipeline uses a 1-100Hz bandpass filter. This is because 1), ICA decompositions
# are more stable when low frequency drifts are removed, and 2) the ICLabel classifier
# is trained on data that has been filtered between 1-100Hz. A notch filter can also be
# optionally specified.
pipeline.config["filtering"]["notch_filter_args"]["freqs"] = [50]
pipeline.filter()

# %%
# Find Nearest Neighbours & return Maximum Correlation
# ----------------------------------------------------
#
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Nearest_neighbors.png
#    :align: center
#    :alt: Nearest Neighbors graphic.
#
#    Nearest Neighbors. The figure shows the steps for finding nearest neighbors. See the text below
#    for descriptions of mathematical notation.
#

# %%
# Whereas :ref:`noisy_sensors` and :ref:`noisy_epochs` operated on a 2D matrix of
# standard deviation values, The next few steps will operate on correlation
# coefficients. Here we describe the procedure for defining the 2D matrix of correlation
# coefficients.

# %%
from pylossless.pipeline import chan_neighbour_r

# %%
#
# Notice that our flagged epochs are dropped.
epochs = pipeline.get_epochs()

# %%
# a) Calculate Correlation Coefficients between each Sensor and its neighboring eighbors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - For each good sensor $i$ in :math:`S_{\mathcal{G}}`, we select its :math:`N` nearest
#   neighbors. I.e. the :math:`N` sensors that are closest to it.
#
# - We call the sensor :math:`i` the *origin*, and its nearest neighbors :math`\hat{s_l}`
#   with :math:`l \in \{1, 2, \ldots, N\}`
#
# - Then, for each epoch :math:`j`, we calculate the correlation coefficient
#   :math:`\rho^t_{(i,\hat{s_l}),j}` between origin sensor :math:`i` and each neighbor
#   :math:`\hat{s_l}` across dimension :math:`t` (samples), returning a 3D matrice of
#   correlation coefficients:
#
# .. math::
#    \mathrm{P}^t = \{\rho^t_{(i, \hat{s_l}),j}\} \in \mathbb{R}^{S_G \times E_G \times n}
#
# Finally, we select the maximum correlation coefficient across the neighbor dimension
# :math:`n`:
#
# .. math::
#    \mathrm{P}^{t,{\text{max}}^n}= \max\limits_{\hat{s_l}}  \rho^t_{(i, \hat{s_l}),j}
#
# Returning a 2D matrix where each value at :math:`(i, j)` is the maximum correlation
# coefficient between sensor :math:`i` and its :math:`N` nearest neighbors, at each epoch
# :math:`j`

# %%
data_r_ch = chan_neighbour_r(epochs, nneigbr=3, method="max")
# maximum correlation out of correlations between ch and its 3 neighbors
data_r_ch

# %%
# This matrix :math:`\mathrm{P}^{t,{\text{max}}^n}`  will be used in the steps below.

# %%
# Flag Bridged Sensors
# --------------------
#
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Flag_bridged_sensors.png
#    :align: center
#    :alt: Flag Bridged Sensors graphic.
#
#    Flag Bridged Sensors. The figure shows the steps for flagging bridged sensors.
#    See the text below for descriptions of mathematical notation.
#

# %%
# a) Calculate the 50th, 75th quantile and IQR across epochs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. math::
#    IQR^e = \mathrm{P}^{t,{\text{max}}^nQ75^e} - \mathrm{P}^{t,{\text{max}}^nQ25^e}
#
# For each sensor, divide the median across epochs by the IQR across epochs. Bridged
# channels should have a high median correlation but a low IQR of the correlation.
# We call this measure the bridge-indicator.
#
# .. math::
#    \mathcal{B}_s = \frac{\mathrm{P}^{t,{\text{max}}^nQ50^e}}{IQR^e}
#
# b) Define a bridging threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, take the 25th, 50th, and 75th quantile of :math:`\mathcal{B}_s` across sensors,
# And calculate the :math:`IQR^s`. A channel :math:`i` is bridged if
#
# .. math::
#    \mathcal{B}_i > B^{Q50^s} +k \times IQR^s
#

# %%
import scipy
from functools import partial

# %%
msr = data_r_ch.median("epoch") / data_r_ch.reduce(scipy.stats.iqr, dim="epoch")
# msr is a 1D vector of size n_sensors
config_trim = 40
config_bridge_z = 6
#
trim = config_trim
if trim >= 1:
    trim /= 100
trim /= 2  # .20 and will be used as (.20, .20)
#
trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))
#
z_val = config_bridge_z  # 6
mask = msr > msr.reduce(trim_mean, dim="ch") + z_val * msr.reduce(
    trim_std, dim="ch"
)  # bridged chans
#
bridged_ch_names = data_r_ch.ch.values[mask]
bridged_ch_names

# %%
# Let's add the outlier channels to our flags
bad_chs = bridged_ch_names
pipeline.flags["ch"].add_flag_cat(kind="bridged", bad_ch_names=bad_chs)
pipeline.flags["ch"]

# %%
# Identify the Rank Channel
# -------------------------
#
# Because the pipeline uses an average reference before the ICA decomposition, it is
# necessary to account for rank deficiency (i.e., every sensor in the montage is
# linearly dependent on the other channels due to the common average reference). To
# account for this, the pipeline flags the sensor (out of the remaining good sensors)
# with the highest median of the max correlation coefficient with its neighbors
# (across epochs):
#
# .. math::
#    \begin{equation}
#    i = \text{arg}\max\limits_i \rho_{i}^{t,{\text{max}}^n,median^j}
#    \end{equation}
#
# This sensor has the least unique time-series out of the remaining set of good sensors
# :math:`S_\mathcal{G}` and is flagged by the pipeline as ``”rank”``. Note that this
# sensor is not flagged because it contains artifact, but only because one of the
# remaining sensors needs to be removed to address rank deficiency before ICA
# decomposition is performed. By choosing this sensor, we are likely to lose little
# information because of its high correlation with its neighbors. This sensor can be
# reintroduced after the ICA has been applied for artifact corrections.

# %%
good_chs = [
    ch for ch in data_r_ch.ch.values if ch not in pipeline.flags["ch"].get_flagged()
]
data_r_ch_good = data_r_ch.sel(ch=good_chs)

flag_ch = [str(data_r_ch_good.median("epoch").idxmax(dim="ch").to_numpy())]
pipeline.flags["ch"].add_flag_cat(kind="rank", bad_ch_names=flag_ch)
pipeline.flags["ch"]

# %%
# Flag low correlation Epochs
# ---------------------------
#
# This step is designed to identify time periods in which many sensors are
# uncorrelated with neighboring sensors. It is similar to the :ref:`noisy_sensors` step,
#
# Again we calculate the 25th and 50th quantile
# of :math:`\mathrm{P}^{t,{\text{max}}^n}`, across the epochs dimension, and calculate
# the lower quantile range :math:`LQR^s`. This results in vectors
# :math:`\mathrm{P}^{t,{\text{max}}^nQ25^e}` and
# :math:`\mathrm{P}^{t,{\text{max}}^nQ50^e}` of size :math:`S_\mathcal{G}`. As for previous
# steps, we define  sensor-specific thresholds for flagging epochs:
#
# .. math::
#    \begin{equation}
#    \tau^e = \mathrm{P}^{t,{\text{max}}^nQ50^e} - LQR^e\times k
#    \end{equation}
#
# And the corresponding indicator matrix:
#
# .. math::
#    \begin{equation}
#    c_{ij} =
#    \begin{cases}
#    1 & \text{if } \rho^{t,{\text{max}}^n}_{ij} < \tau^e_i \\
#    0 & \text{if } \rho^{t,{\text{max}}^n}_{ij} \geq \tau^e_i
#    \end{cases}
#    \end{equation}
#
# We average the indicator matrix across sensors and obtain a vector :math:`C^{\mu_s}`
# that we use to flag uncorrelated epochs using the following criterion:
#
# .. math::
#    c^{\mu_e}_i > \tau^{p}.
#

# %%
# Step a
q25, q50 = data_r_ch.quantile([0.25, 0.5], dim="epoch")
#
# Define the LQR
lqr = q50 - q25
#
# define a threshold
k = 3
lower_threshold = q50 - lqr * k
#
outlier_mask = data_r_ch < lower_threshold
#
percent_outliers = outlier_mask.astype(float).mean("ch")
#
p_threshold = 0.2
bad_epochs = percent_outliers[percent_outliers > p_threshold].coords.to_index().values
#
# Add the outlier epochs to our flags
pipeline.flags["epoch"].add_flag_cat(
    kind="uncorrelated", bad_epoch_inds=bad_epochs, epochs=epochs
)
pipeline.raw.annotations.description

# %%
# in this case, no epochs were flagged as uncorrelated.
#

# %%
# Flag low correlation Sensors
# -----------------------------
#
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Flag_uncorrelated_sensors.png
#    :align: center
#    :alt: Flag Uncorrelated Sensors graphic.
#
#    Flag Uncorrelated Sensors. The figure shows the steps for flagging uncorrelated
#    sensors. See the text below for descriptions of mathematical notation.
#
# This step is designed to identify sensors that have an unusually low correlation with
# neighboring sensors. The operations involved by this step are similar to those of the
# :ref:`noisy_sensors` step, except we use maximal nearest neighbor correlations instead
# of dispersion and the left instead of the right tail of the distribution to set
# the threshold for outliers.

# %%
# a) Take lower quantile range and defined sensor-specific thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We get the indicator matrix as described previously, using
#
# .. math::
#    \tau^e_i = \mathrm{P}^{t,{\text{max}}^nQ50^e} - LQR^e\times k
#
# and
#
# .. math::
#    c_{ij} =
#    \begin{cases}
#    1 & \text{if } \rho^{t,{\text{max}}^n}_{ij} < \tau^e_i \\
#    0 & \text{if } \rho^{t,{\text{max}}^n}_{ij} \geq \tau^e_i
#    \end{cases}

# %%
# b) Identify uncorrelated sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We define a threshold as we did in the previous step and flag uncorrelated epochs
# :math:`j` if :math:`c^{\mu_s}_j > \tau^{p}`.

# %%
q25, q50 = data_r_ch.quantile([0.25, 0.5], dim="ch")
#
# Define LQR
lqr = q50 - q25
#
# define a threshold
k = 3
lower_threshold = q50 - lqr * k
#
# Identify correlations less than the threshold
outlier_mask = data_r_ch < lower_threshold
percent_outliers = outlier_mask.astype(float).mean("epoch")
#
p_threshold = 0.2
bad_chs = percent_outliers[percent_outliers > p_threshold].coords.to_index().values
#
# Add the outlier channels to our flags
pipeline.flags["ch"].add_flag_cat(kind="uncorrelated", bad_ch_names=bad_chs)
pipeline.flags["ch"]

# %%
# In this case, no sensors were flagged as uncorrelated.

# %%
# Run Initial ICA
# ---------------
#
# The pipeline by runs ICA two times. The first ICA is only used to identify
# noisy periods in its IC activation time-series. For this reason, the pipeline
# uses the FastICA algorithm for speed.

# %%
pipeline.run_ica("run1")

# %%
# Flag Noisy IC Activation time-periods
# -------------------------------------
#
# This step follows the same procedure as the :ref:`noisy_sensors` step, except that
# the data is now the IC activation time-series. thus we start with a 3D matrix
# :math:`X_{ica} \in \mathbb{R}^{I_\mathcal{G} \times E_\mathcal{G} \times T}` of
# IC time-courses rather than scalp EEG data and where :math:`I` is the set of
# independent components.
#

# %%
pipeline.flag_noisy_ics()

# %%
pipeline.raw.annotations.description

# %%
# Run Final ICA
# -------------
#
# Now The pipeline runs the final ICA decomposition, this time using the extended
# Infomax algorithm. Note that any sensors or time-periods that have been flagged
# up to this point will not be passed into the ICA decomposition. For the sake of
# time, we will not run the second ICA here, as there are no more pipeline calculations.

# %%
# Run ICLabel Classifier
# ----------------------
#
# The pipeline will run the ICLabel classifier on the final ICA, which will produce a
# label for each IC, one of ``"brain"``, ``"muscle"``, ``"eog"`` (eye), ``"ecg"``
# (heart), ``line_noise``, or ``"channel_noise"``.
#
#
# Conclusion
# ----------
# And that's all! See the other pylossless tutorials for brief examples on running the
# pipeline on your own data, and rejecting the flagged data.
#
