Run ICA
=======

ICA is run twice in the pipeline. The first time IC decomposition is simply used to
detect time periods in which the IC time courses are noisy. The second ICA is the
final decomposition, which is passed to the ICLabel classifier.

There are two YAML fields for tuning ICA parameters. the ``ica`` field will contain
parameters for detecting noisy time periods in the data with outlying IC time courses,
similar to the approach in the :ref:`noisy_epochs` step. The ``ica_args`` field is
where you should pass keyword arguments that are accepted by MNE-Pythons :class:`mne.preprocessing.ICA`
class.

For a detailed description of the ``ica`` field parameters, please see the :ref:`noisy_epochs`
step, as the approach and definitions are identical.

.. code-block:: yaml

    ica:
        flag_crit: 0.2
        outlier_method: quantile
        outliers_kwargs:
            k: 6
            lower: 0.25
            upper: 0.75

    # See arguments definition from mne.preprocessing.ICA
    ica_args:
        run1:
            method: fastica
        run2:
            method: infomax
        fit_params:
            extended: True
