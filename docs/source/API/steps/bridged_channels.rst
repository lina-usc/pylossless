Identify Bridged Channels
=========================

The ``bridged_channels`` step identifies and marks EEG channels as bridged if
their per-epoch correlation coefficient with neighboring channels (computed across samples)
is consistently a high outlier relative to the corresponding coefficients of other channels.

Examples
--------

.. code-block:: yaml

    bridged_channels:
        bridge_trim: 40
        bridge_z: 6

Parameters
----------

- ``bridge_trim`` : float, default=0.2
    When taking the trimmed mean and trimmed standard deviation across channels, what percent
    should be trimmed the data. Default is ``0.2``, which means that 10% will be trimmed from
    each tail.

- ``bridge_z`` : float, default=6
    Scaling factor for threshold detection.
