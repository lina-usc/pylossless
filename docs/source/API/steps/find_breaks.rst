Find Breaks
===========

The ``find_breaks`` step detects and marks breaks between experimental tasks in the data.
By default it is ``null``, meaning no breaks are detected. To enable break detection, set the
``find_breaks`` section in the YAML configuration file with the desired parameters.

Examples
--------

To enable break detection and simply use the default parameters defined in the
:func:`mne.preprocessing.annotate_break` function, you can set the configuration as follows:

.. code-block:: yaml

   find_breaks: {}

To customize the break detection parameters, you can specify them in the
``find_breaks`` section. Valid parameters include those accepted by the
:func:`mne.preprocessing.annotate_break` function:

.. code-block:: yaml

   find_breaks:
    min_break_duration: 20.0 # Minimum duration of a break in seconds


Notes
-----

The ``find_breaks`` step will add ``"BAD_break"`` annotations to the raw data
object for each detected break. These annotations can then be used in subsequent
steps of the pipeline to exclude break periods from processing.