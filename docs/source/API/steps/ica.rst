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


.. _ica-random-seeds:

Reproducibility and random seeds
--------------------------------

PyLossless uses the top-level ``random_seed`` configuration value for stochastic
pipeline steps. Both ICA runs inherit this value when their run-specific
``random_state`` is omitted. The default is ``97``, preserving the behavior of
older PyLossless versions::

    random_seed: 97
    ica:
      ica_args:
        run1:
          method: fastica
        run2:
          method: infomax
          fit_params:
            extended: true

Set ``random_seed: null`` to let the ICA implementation initialize
non-deterministically. For a deliberate per-run override, set
``random_state`` inside that run. The local value takes precedence::

    random_seed: 97
    ica:
      ica_args:
        run1:
          method: fastica
        run2:
          method: infomax
          random_state: 42
          fit_params:
            extended: true

For reproducible analyses, save the resolved configuration with the derivative
and keep the software environment fixed. A seed controls pseudo-random
initialization; it does not guarantee bitwise-identical results across different
library versions, numerical backends, hardware, or thread configurations.
