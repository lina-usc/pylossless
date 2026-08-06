Identify Noisy Epochs
=====================

The ``noisy_epochs`` step identifies and marks EEG epochs as *noisy* based on
whether their per-channel signal standard deviation is consistently an outlier
relative to other epochs.

This step operates in two stages:

1. For each epoch, channels whose standard deviation is identified as an outlier
   are flagged.
2. An epoch is marked as noisy if it more than ``flag_crit``
   proportion of channels were flagged for this epoch.

Examples
--------

.. code-block:: yaml

    noisy_epochs:
        flag_crit: 0.2
        outlier_method: quantile
        outliers_kwargs:
            k: 6
            lower: 0.25
            upper: 0.75

Parameters
----------

- ``flag_crit`` : float, default=0.2
    Proportion of channels that must be flagged as outliers in an epoch
    for the epoch to be marked as noisy.

- ``outlier_method`` : str, default='quantile'
    Method used to identify outlier epochs. Options are
    ``'quantile'``, ``'trimmed'``, or ``'fixed'``.

    - ``'quantile'``
        Identifies outliers using asymmetric, quantile-based thresholds.
        For a given epoch, channels are flagged if their standard deviation
        lies outside

        ``μ - k * (μ - q_low)`` or ``μ + k * (q_high - μ)``

        where ``μ`` is the mean standard deviation across channels, ``q_low`` and
        ``q_high`` are the lower or upper quantile specified by
        ``outliers_kwargs['lower']`` and ``outliers_kwargs['upper']``, and ``k`` is a
        scaling factor.

        This method is robust to skewed distributions and is suitable when
        channel variability is not symmetrically distributed.

    - ``'trimmed'``
        Identifies outliers using robust statistics by computing the trimmed mean
        and trimmed standard deviation across channels. For a given epoch,
        channels with standard deviation values outside

        ``trimmed_mean ± k * trimmed_std``

        are flagged as noisy.

        This method reduces the influence of extreme artifacts while preserving
        sensitivity to moderately noisy channels.

    - ``'fixed'``
        Identifies outliers using fixed, absolute thresholds. For a given epoch,
        channels with standard deviation values outside the thresholds (in volts)
        specified by ``outliers_kwargs['lower']`` and ``outliers_kwargs['upper']``
        are flagged.

- ``outliers_kwargs`` : dict
    Additional keyword arguments specific to the chosen outlier detection method.

    - For ``'quantile'``:
        - ``k`` : float, default=6
            Scaling factor for the quantile-based thresholds.
        - ``lower`` : float, default=0.25
            Lower quantile.
        - ``upper`` : float, default=0.75
            Upper quantile.

    - For ``'trimmed'``:
        - ``k`` : float, default=3
            Scaling factor for trimmed standard deviation thresholds.

Notes
-----

The ``trimmed`` and ``fixed`` methods are not robustly tested in the current codebase.
Please use with caution and validate results carefully.