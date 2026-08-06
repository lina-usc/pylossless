Identify Neighboring sensors
============================

There are a few pipeline steps (e.g. identifying uncorrelated and bridged channels) that
require information about which sensors are neighbors to one another. This step provides
parameters for fine tuning neighbor identification.


Example
+++++++

.. code-block:: yaml

    nearest_neighbors:
        n_nbr_ch: 3


Parameters
----------

- ``n_nbr_ch``: int, default=3

    The number of neighboring sensors you wish to identify.