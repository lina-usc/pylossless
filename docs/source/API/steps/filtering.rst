Filtering
=========

The ``filtering`` step applies band-pass, high-pass, and low-pass filters to the data.
In the YAML file, the ``filtering`` section should be at the top level. Valid parameters
for the ``filtering`` step are ``filter_args`` and ``notch_filter_args``:

- ``filter_args``: A dictionary of keyword arguments to pass to the
  :meth:`mne.io.Raw.filter` method for band-pass, high-pass, or low-pass filtering.
  See the MNE-Python documentation for details on available parameters.
- ``notch_filter_args``: A dictionary of keyword arguments to pass to the
  :meth:`mne.io.Raw.notch_filter` method for notch filtering. See the MNE-Python
  documentation for details on available parameters.


Example
-------

.. code-block:: yaml

   filtering:
    filter_args:
        l_freq: 1.0        # High-pass filter cutoff frequency in Hz
        h_freq: 100.0       # Low-pass filter cutoff frequency in Hz
        verbose: 'WARNING'
    notch_filter_args:
        freqs: [60]
    

Note
----

By default, PyLossless runs MNE-ICALabel, which strongly recommends that the
data be filtered between 1 and 100 Hz. If you set a different filter range here, be
aware that MNE-ICALabel will trigger a warning.
