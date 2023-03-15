# PyLossless EEG Processing Pipeline

**Note: This repository is in a constant state of flux and not yet ready for outside use!**

## Motivation

This project is a port of the MATLAB Lossless EEG Processing Pipeline ([Github repo](https://github.com/BUCANL/EEG-IP-L)) presented in [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848). This port aims at 1) making this pipeline available to the Python community and 2) providing a version of the pipeline that is easier to deploy by outsiders.

This pipeline is built on the idea that sharing and pooling data across the scientific community is most efficient when sharing a standardized (e.g., in [BIDS](https://www.nature.com/articles/s41597-019-0104-8)) and "clean" version of a dataset. However, cleaning artifacts in a dataset generally results in a loss of data (i.e., the original recorded signals are generally not recoverable). This is particularly problematic given that preprocessing steps for a dataset are rarely perfect (i.e., future developments may offer methods that would perform better at removing some artifacts) and can be project-dependent. The Lossless pipeline addresses this issue by proposing a "lossless" process where data are annotated for artifacts in a non-destructive way, so that users have access to a readily clean dataset if they are comfortable with the existing annotations. Alternative, they can choose which annotations to use for preprocessing in a piecemeal fashion, or simply use the raw data without excluding any artifacts based on provided annotations. Artifacts are annotated for channels, epochs, and independent components; see  [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848) for a more detailed presentation.


## 📘 Installation and usage instructions

Please find the documentation at
[**pylossless.readthedocs.io**](https://pylossless.readthedocs.io/en/latest/index.html).