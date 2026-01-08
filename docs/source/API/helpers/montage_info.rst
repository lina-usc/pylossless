Setting the EEG Montage
=======================


There are a number of pipeline operations that requires information
about the sensor locations on the scalp (e.g. sensor locations are required for
identifying neighboring sensors). Under the hood, the pipeline uses MNE-Python
to set the EEG montage.

There are 3 separate fields that can be used for controlling montage setting
in the pipeline. all 3 of these fields exist under the ``project`` field of the
configuration file/class.

If your data are not already BIDS compliant, and you plan to use the pipeline
to bidfify your data, then you will want to set the ``bids_montage`` field. This will
set the montage of your EEG data as part of the BIDSification process, and will write
the data to an ``electrodes.tsv`` file, along with the BIDSified EEG data. In this case,
later steps in the pipeline will just use the aforementioned ``electrodes.tsv`` file for
retrieving information about sensor locations, and you don't have to set a value in the
``analysis_montage`` field.

If your data are already BIDS compliant, you can also leave ``analysis_montage`` field
blank, assuming there is an ``electrodes.tsv`` file for the pipeline to use.

If you wish to override this and use a specific montage, you are also allowed to
use one of the available montages in MNE-Python. You can find a list of options by
inspecting the output of :func:`mne.channels.get_builtin_montages`.

If your EEG data has a custom montage that cannot be used with one of the outputs
of ``get_builtin_montages``, you should set your montage and BIDSify your data before
running the pipeline.

Finally, the ``set_montage_kwargs`` field will accept any parameter that is accepted
by MNE-Python's :func:`mne.Info.set_montage`.

Example
+++++++

.. code-block:: yaml

    project:
        readme: "# Description of the dataset"

        # Montage use to make file BIDS compliant.
        # Can be path to digitized montage OR a string of one of mne's built in
        # standard montages as specified by mne.channels.get_builtin_montages().
        # Can be left empty if the input dataset is already in BIDS format.
        bids_montage: GSN-HydroCel-129

        # montage used while running the lossless pipeline.
        # if empty, the pipeline will use the electrodes.tsv sidecar file, if created
        # during the BIDS conversion.
        # If specified, needs to be a string of one of mne's built in standard montages.
        analysis_montage: ""

        set_montage_kwargs: {}

