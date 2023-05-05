.. pyLossless documentation master file, created by
   sphinx-quickstart on Fri Jan  6 12:24:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyLossless EEG Processing Pipeline
==================================
   
.. toctree::
   :maxdepth: 1
   :hidden:

   install
   implementation
   API/API_index
   generated/index
   contributing



.. grid::
   
   .. grid-item-card::

      |:mechanical_arm:| Automated
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Automatic Processing Pipeline
      cleans your EEG data. |:broom:|


   .. grid-item-card::

      |:snake:| Built on Python
      ^^^^^^^^^^^^^^^^^^^^^^^^^
      Ported from MATLAB for easier
      use and access!

   .. grid-item-card::

      |:recycle:| Non-destructive
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Keeps your EEG continuous, so
      you can epoch your data however
      and whenever you want to.

.. grid::

   .. grid-item-card:: 

      |:pencil:| Artifacts are Noted
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Bad channels, times, and
      components are stored as
      ``Annotations`` in your
      raw data. 

   .. grid-item-card::

      |:woman_technologist:| Streamlined Review
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Web dashboard built with Plotly/Dash helps
      you Review the pipeline output and make
      informed decisions about your data
      
.. image:: ./_images/qc_screenshot.png
   :alt: pylossless-qc-dashboard-screenshot
   :align: center