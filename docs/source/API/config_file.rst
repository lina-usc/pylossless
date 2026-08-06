Pipeline Configuration File
===========================

The configuration file for PyLossess is a YAML file that specifies various parameters
and settings for the processing steps. The file can be read and parsed when instantiating the
the `class:~pyLossless.config.Config` class, e.g.
``pylossless.config.Config("path/to/my/config_file.yaml")``. If you don't already have a configuration
file, you can instantiate an empty configuration class (e.g. ``pylossless.config.Config()``)
and then use the `meth:~pylossless.config.Config.load_default` method to initialize the
pipeline with default settings.

If you load the default configuration, you will most likely want to modify some of the settings
before running the pipeline. You can do this by either editing the configuration file directly
or by modifying the attributes of the `class:~pyLossless.config.Config` instance in your Python code.
To save the configuration instance back to a YAML file (for parameter editing or for later use),
you can use the `meth:~pylossless.config.Config.save` method, e.g.
``config_instance.save("path/to/my/modified_config_file.yaml")``.

The configuration file includes settings for various steps in the PyLossless pipeline, but also
includes settings for describing metadata about the dataset, and settings for passing arguments to
third-party dependencies (e.g. MNE-Python).

To see examples of the Configuration files, see below:

- `Adult Example <https://github.com/lina-usc/pylossless/blob/main/pylossless/assets/ll_default_config_adults.yaml>`_
- `Infant Example <https://github.com/lina-usc/pylossless/blob/main/pylossless/assets/ll_default_config_infants.yaml>`_

Configuring Pipeline Steps
--------------------------

Each processing step in the PyLossless pipeline has its own section in the configuration file. Each section
includes parameters that control the behavior of that step. For example, the `preprocessing` section includes
parameters for filtering, resampling, and artifact rejection. The `source_reconstruction` section includes
parameters for selecting the source space, forward model, and inverse method.


Pipeline steps
++++++++++++++

+--------------------------+--------------------------------------------------------------+
| Step Name                | Description                                                  |
+==========================+==============================================================+
| `filtering`_             | Parameters for band-pass, high-pass, and low-pass filtering  |
|                          | of the data.                                                 |
+--------------------------+--------------------------------------------------------------+
| `find_breaks`_           | Parameters for detecting and marking breaks between          |
|                          | experimental tasks.                                          |
+--------------------------+--------------------------------------------------------------+
| `noisy_channels`_        | Parameters for identifying and marking noisy channels in     |
|                          | the data.                                                    |
+--------------------------+--------------------------------------------------------------+
| `noisy_epochs`_          | Parameters for identifying and marking noisy time periods in |
|                          | the data.                                                    |
+--------------------------+--------------------------------------------------------------+
| `uncorrelated_channels`_ | Parameters for identifying and marking uncorrelated channels |
|                          | in the data.                                                 |
+--------------------------+--------------------------------------------------------------+
| `uncorrelated_epochs`_   | Parameters for identifying and marking uncorrelated time     |
|                          | periods in the data.                                         |
+--------------------------+--------------------------------------------------------------+
| `bridged_channels`_      | Parameters for identifying and marking bridged channels in   |
|                          | the data.                                                    |
+--------------------------+--------------------------------------------------------------+
| `ica`_                   | Parameters for performing ICA to identify and remove         |
|                          | artifacts from the data.                                     |
+--------------------------+--------------------------------------------------------------+

.. _filtering: steps/filtering.html

.. _find_breaks: steps/find_breaks.html

.. _noisy_channels: steps/noisy_channels.html

.. _noisy_epochs: steps/noisy_epochs.html

.. _uncorrelated_epochs: steps/uncorrelated_epochs.html

.. _uncorrelated_channels: steps/uncorrelated_channels.html

.. _bridged_channels: steps/bridged_channels.html

.. _ica: steps/ica.html

Pipeline helper configurations
++++++++++++++++++++++++++++++

+-----------------------+---------------------------------------------------------------------------+
| Helper Name           | Description                                                               |
+=======================+===========================================================================+
| `epoching`_           | Parameters for specifying how to epoch the continuous data into segments. |
+-----------------------+---------------------------------------------------------------------------+
| `nearest_neighbors`_  | Parameters for specifying sensor neighbors.                               |
+-----------------------+---------------------------------------------------------------------------+
| `montage_info`_       | Parameters for specifying sensor montage information.                     |
+-----------------------+---------------------------------------------------------------------------+

.. _epoching: helpers/epoching.html

.. _nearest_neighbors: helpers/nearest_neighbors.html

.. _montage_info: helpers/montage_info.html

