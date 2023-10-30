r"""
.. _notation:

PyLossless Calculations: Understanding each step
================================================

This tutorial explains the calculations that PyLossless performs at each step of the
pipeline. We will use example EEG data to demonstrate the
calculations.

Notation 
--------
Before we begin, we define some notation that will be used throughout the text:

- ``s``, ``e``, and ``t`` are sensor, epochs, and time dimensions, respectively.

- a 3D matrix :math:`X \in \mathbb{R}^{S_\mathcal{G} \times E_\mathcal{G} \times T}`,
  where :math:`S_\mathcal{G}` and :math:`E_\mathcal{G}` are the sets of good sensors and
  epochs, respectively, and :math:`T`, is the number of time points (i.e. samples).

- We use superscripts to denote operations across a dimension, and we use subscripts to
  denote indexing a dimension.

- a single sensor :math:`i` as
  :math:`X\big|_{s=i} \in \mathbb{R}^{E_\mathcal{G} \times T}`,
  with :math:`i \in S_\mathcal{G}`.

- a single epoch :math:`j` as
  :math:`X\big|_{e=j} \in \mathbb{R}^{S_\mathcal{G} \times T}`,
  with :math:`j \in E_\mathcal{G}`.

- sensor-specific thresholds for rejecting epochs as
  :math:`\tau^e_i \in \mathbb{R}^{S_\mathcal{G}}`

- epoch-specific thresholds for rejecting sensors as
  :math:`\tau^s_j \in \mathbb{R}^{E_\mathcal{G}}`

- *quantiles* as :math:`Q\#^{dim}`: i.e. :math:`Q75^s` is the 75th *quantile* along
  the sensor dimension. The function :math:`Q75^s(X)` computes the 75th quantile along
  the :math:`s` dimension of matrix :math:`X`, resulting in a matrix noted
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

# Load a default configuration file
config = ll.config.Config()
config.load_default()
config.save("test_config.yaml")

# Create a pipeline instance
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
epochs = pipeline.get_epochs()

# %%
#
# Let's convert our epochs object into a named :class:`xarray.DataArray` object.
from pylossless.pipeline import epochs_to_xr

#
epochs_xr = epochs_to_xr(epochs, kind="ch")
epochs_xr  # 277 epochs, 50 sensors, 602 samples per epoch

# %%
# Flag Noisy Sensors
# ------------------
# .. figure:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/Flag_noisy_sensors.png
#    :align: center
#    :alt: Flag Noisy Sensors graphic.
#
#    Flag Noisy Sensors. The figure shows the steps for flagging noisy sensors. See the text below
#    for descriptions of mathematical notation.

# %%
# First we take standard deviation of
# :math:`X \in \mathbb{R}^{S \times E \times T}` across the samples dimension :math:`t`
# resulting in a 2D matrix :math:`X^{\sigma_{t}} \in \mathbb{R}^{S \times E}`
trim_ch_sd = epochs_xr.std("time")
trim_ch_sd

# %%
# Take the 50th and 75th quantile across dimension sensor of :math:`X^{\sigma_{t}}`
# ---------------------------------------------------------------------------------
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
# -----------------------------------------------------------
#
# .. math::
#    UQR^s = X^{{\sigma}_T{Q75}^s} - X^{{\sigma}_T{Q50}^s}
#
# This operation results in a 1D vector of size :math:`E`.
uqr = q75 - q50
uqr

# %%
# Identify outlier Indices :math:`(i, j)`
# ---------------------------------------
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
# ## c) Identify noisy sensors part 1
# To identify outlier sensors, we average across the epoch dimension of our indicator
# matrix :math:`C` and obtain :math:`C^{\mu_e} \in \mathbb{R}^{S_\mathcal{G}}`, which
# is a vector of fractional numbers :math:`c^{\mu_e}_i` representing the percentage of
# epochs for which that sensor is an outlier.
percent_outliers = outlier_mask.astype(float).mean("epoch")
percent_outliers  # percent of epochs that sensor is an outlier

# %%
# d) Identify noisy sensors part 2
# --------------------------------
# Next, we define a fractional threshold :math:`\tau^{p}` (:math:`p` for percentile;
# default, ``.20``) as a cutoff point for determining if a sensor should be marked
# artifactual. The sensor :math:`i` is flagged as noisy if
# :math:`c^{\mu_e}_i > \tau^{p}`.
threshold = 0.2
percent_outliers[percent_outliers > threshold]  # noisy sensors
