.. pyLossless documentation master file, created by
   sphinx-quickstart on Fri Jan  6 12:24:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyLossless EEG Processing Pipeline
==================================

.. note::
   This repository is in a constant state of flux and not yet ready for outside use!
   
.. toctree::
   :maxdepth: 1
   :hidden:

   install
   implementation
   API/API_index
   generated/index
   contributing


Motivation
==================
This project is a port of the MATLAB Lossless EEG Processing Pipeline
`(Github repo) <https://github.com/BUCANL/EEG-IP-L>`_ presented in `Desjardins
et al
(2021) <https://www.sciencedirect.com/science/article/pii/S0165027020303848>`_.
This port aims at 1) making this pipeline available to the Python community
and 2) providing a version of the pipeline that is easier to deploy by
outsiders.

This pipeline is built on the idea that sharing and pooling data across the
scientific community is most efficient when sharing a standardized
(e.g., in BIDS) and "clean" version of a dataset. However, cleaning artifacts
in a dataset generally results in a loss of data (i.e., the original recorded
signals are generally not recoverable). This is particularly problematic given
that preprocessing steps for a dataset are rarely perfect (i.e., future
developments may offer methods that would perform better at removing some
artifacts) and can be project-dependent. The Lossless pipeline addresses this
issue by proposing a "lossless" process where data are annotated for artifacts
in a non-destructive way, so that users have access to a readily clean dataset
if they are comfortable with the existing annotations. Alternative, they can
choose which annotations to use for preprocessing in a piecemeal fashion, or
simply use the raw data without excluding any artifacts based on provided
annotations. Artifacts are annotated for channels, epochs, and independent
components; see Desjardins et al (2021) for a more detailed presentation.