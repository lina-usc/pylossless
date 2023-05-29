[![codecov](https://codecov.io/github/lina-usc/pylossless/branch/main/graph/badge.svg?token=SVAD8HTJNG)](https://codecov.io/github/lina-usc/pylossless)

![logo](./docs/source/_static/logo_white.png)

![QCR Dashboard](./docs/source/_images/qc_screenshot.png)

### **Note: This software has alpha status. This means that this package is young and will likely undergo frequent changes and improvements.


## 📘 Installation and usage instructions

This package can be install from PyPI with
```bash
$ pip install pylossless
```

The development version can be installed from GitHub with
```bash
$ git clone git@github.com:lina-usc/pylossless.git
$ pip install --editable ./pylossless
```
for an editable installation, or simply with 
```bash
$ pip install git+https://github.com/lina-usc/pylossless.git
```
for a static version. 

Please find the full documentation at
[**pylossless.readthedocs.io**](https://pylossless.readthedocs.io/en/latest/index.html).


## ▶️ Running the pyLossless Pipeline
Below is a minimal example that runs the pipeline one of MNE's sample files.  
```
import pylossless as ll 
import mne
fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' /  'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

config = ll.config.Config()
config.load_default()
config.save("my_project_ll_config.yaml")

pipeline = ll.LosslessPipeline('my_project_ll_config.yaml')
pipeline.run_with_raw(raw)
```

Once it is completed, You can see what channels and times were flagged:
```
print(pipeline.flagged_chs)
print(pipeline.flagged_epochs)
```

Once you are ready, you can save your file:
```
pipeline.save(pipeline.get_derivative_path(bids_path), overwrite=True)
```

## ▶️ Example HPC Environment Setup

Assuming you are on a system such as [Narval](https://docs.alliancecan.ca/wiki/Narval/en):

```bash
module load python/3.10

# Build the virtualenv in your homedir
virtualenv --no-download eeg-env
source eeg-env/bin/activate

pip install --no-index mne
pip install --no-index pandas
pip install --no-index xarray
pip install --no-index pyyaml
pip install --no-index sklearn
pip install mne_bids

# Clone down mne-iclabel and switch to the right version and install it locally
git clone https://github.com/mne-tools/mne-icalabel.git
cd mne-icalabel
git checkout maint/0.4
pip install .

# Clone down pipeline and install without reading dependencies
git clone git@github.com:lina-usc/pylossless.git
cd pylossless
pip install --no-deps .

# Verify that the package has installed correct with an import
python -c 'import pylossless'
```

## 👩‍💻 Dashboard Review
[![Open in Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/lina-usc/pylossless/blob/main/notebooks/qc_example.ipynb)

![QCR Dashboard](./docs/source/_images/qc_screenshot.png)

After running the Lossless pipeline, you can launch the Quality Control
Review (QC) dashboard to review the pipeline's decisions on each file!
You can flag additional channels, times and components, and edit flags
made by the pipeline.

First install the dashboard requirements
```bash
$ cd ./path/to/pylossless/on/your/computer
$ pip install --editable .[dash]
```

```bash
$ python pylossless/dash/app.py
```

## Motivation

This project is a port of the MATLAB Lossless EEG Processing Pipeline ([Github repo](https://github.com/BUCANL/EEG-IP-L)) presented in [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848). This port aims at 1) making this pipeline available to the Python community and 2) providing a version of the pipeline that is easier to deploy by outsiders.

This pipeline is built on the idea that sharing and pooling data across the scientific community is most efficient when sharing a standardized (e.g., in [BIDS](https://www.nature.com/articles/s41597-019-0104-8)) and "clean" version of a dataset. However, cleaning artifacts in a dataset generally results in a loss of data (i.e., the original recorded signals are generally not recoverable). This is particularly problematic given that preprocessing steps for a dataset are rarely perfect (i.e., future developments may offer methods that would perform better at removing some artifacts) and can be project-dependent. The Lossless pipeline addresses this issue by proposing a "lossless" process where data are annotated for artifacts in a non-destructive way, so that users have access to a readily clean dataset if they are comfortable with the existing annotations. Alternative, they can choose which annotations to use for preprocessing in a piecemeal fashion, or simply use the raw data without excluding any artifacts based on provided annotations. Artifacts are annotated for channels, epochs, and independent components; see  [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848) for a more detailed presentation.
