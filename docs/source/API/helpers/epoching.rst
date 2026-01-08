Epoching
========

The pipeline typically epochs the raw data at the beginning of every step (e.g. re-referencing
identifying noisy channels and epochs, ICA, etc.). There are two fields in the configuration that
can fine tune the epoching process: ``epoching`` and ``epoch_args``. the ``epoch_args`` field accepts
any parameters that are accepted by MNE-Python's :class:`mne.Epochs` class. The ``epoching`` field
accepts any parameters that are accepted by MNE-Pythons :func:`mne.make_fixed_length_events` function.

By default, the pipeline will epoch the data into 1-second non-overalapping epochs.

Example
+++++++

.. code-block:: yaml

    epoching:
        overlap: 0

    # See arguments definition from mne.Epochs
    epochs_args:
        baseline: null
        tmax: 1
        tmin: 0