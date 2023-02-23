# PyLossless EEG Processing Pipeline

**Note: This repository is in a constant state of flux and not yet ready for outside use!**


## Motivation

This project is a port of the MATLAB Lossless EEG Processing Pipeline ([Github repo](https://github.com/BUCANL/EEG-IP-L)) presented in [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848). This port aims at 1) making this pipeline available to the Python community and 2) providing a version of the pipeline that is easier to deploy by outsiders.

This pipeline is built on the idea that sharing and pooling data across the scientific community is most efficient when sharing a standardized (e.g., in [BIDS](https://www.nature.com/articles/s41597-019-0104-8)) and "clean" version of a dataset. However, cleaning artifacts in a dataset generally results in a loss of data (i.e., the original recorded signals are generally not recoverable). This is particularly problematic given that preprocessing steps for a dataset are rarely perfect (i.e., future developments may offer methods that would perform better at removing some artifacts) and can be project-dependent. The Lossless pipeline addresses this issue by proposing a "lossless" process where data are annotated for artifacts in a non-destructive way, so that users have access to a readily clean dataset if they are comfortable with the existing annotations. Alternative, they can choose which annotations to use for preprocessing in a piecemeal fashion, or simply use the raw data without excluding any artifacts based on provided annotations. Artifacts are annotated for channels, epochs, and independent components; see  [Desjardins et al (2021)](https://www.sciencedirect.com/science/article/pii/S0165027020303848) for a more detailed presentation.


## Installation

This package is not yet deployed on PyPI. It can therefore be installed with

```bash
$ git clone git@github.com:lina-usc/pylossless.git
$ pip install --editable ./pylossless
```
for an editable installation, or simply with 
```bash
$ pip install git+https://github.com/lina-usc/pylossless.git
```
for a static version. 


## Usage 

First importing the package as `ll` for expediency.

```python
import pylossless as ll
```

Running the pipeline always requires 1) a dataset and 2) a configuration file describing the parameters for the preprocessing. A default version of this configuration file can be fetched as a starting point that can be adjusted to the specific needs of a given project

```python
config = ll.config.Config()
config.load_default()
config.print()
config.save("my_project_ll_config.yaml")
```
More information about the description of the different fields can be found [here](./doc/config.md).

The PyLossless pipeline expects the EEG recordings to be stored as BIDS data. We can demonstrate the usage of the pipeline on a BIDS dataset loaded from OpenNeuro. First, we need to download the dataset

```python
#Shamelessly copied from https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
#pip install openneuro-py
import openneuro
from mne.datasets import sample

dataset = 'ds002778'
subject = 'pd6'

# Download one subject's data from each dataset
bids_root = sample.data_path() / dataset
bids_root.mkdir(exist_ok=True)

openneuro.download(dataset=dataset, target_dir=bids_root,
                   include=[f'sub-{subject}'])
```

Now that we have a BIDS dataset saved locally, we can use `mne_bids` to load this dataset as as `mne.io.Raw` instance

```python
import mne_bids

datatype = 'eeg'
session = 'off'
task = 'rest'
suffix = 'eeg'
bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task,
                              suffix=suffix, datatype=datatype, root=bids_root)

raw = mne_bids.read_raw_bids(bids_path)
```

Great! We have our two ingredients (a dataset and a configuration file), and we can now run the pipeline on that dataset (actually, just one recording in that case)

```python
pipeline = ll.LosslessPipeline('my_project_ll_config.yaml')
pipeline.run(raw)

```

Note that running the pipeline for a full dataset is not much more complicated. We only need a list of `BIDSPath` for all the recordings of that dataset. For example, if `bids_paths` contains such a list, the whole dataset can be processed as follows:

```
pipeline.run_dataset(bids_paths)
```

This function essentially loads one raw instance after another from the BIDS recordings specified in `bids_paths` and calls `pipeline.run(raw)` with these raw objects.



## BIDSification

PyLossless provides some functions to help the user import non-BIDS recordings. Since the code to import datasets recorded in different formats and with different properties can vary much from one project to the next, the user must provide a function that can load and return a `raw` object along with the standard MNE `events` array and `event_id` dictionary. For example, in the case of our dataset

```python
# Example of importing function
import tempfile
def egi_import_fct(path_in, stim_channel):

    # read in a file
    raw = mne.io.read_raw_egi(path_in, preload=True)

    # events and event IDs for events sidecar
    events = mne.find_events(raw, stim_channel=['STI 014'])
    event_id = raw.event_id

    # MNE-BIDS doesn't currently support RawMFF objects.
    with tempfile.TemporaryDirectory() as temp_dir:
        raw.save(Path(temp_dir) / "tmp_raw.fif")

        # preload=True is important since this file is volatile
        raw = mne.io.read_raw_fif(Path(temp_dir) / 'tmp_raw.fif', preload=True)

    # we only want EEG channels in the channels sidecar
    raw.pick_types(eeg=True, stim=False)
    raw.rename_channels({'E129': 'Cz'})  # to make compatible with montage

    return raw, events, event_id
```

Then, the dataset can be converted to BIDS as follows

```python
import_args = [{"stim_channel": 'STI 014', "path_in": './sub-s004-ses_07_task-MMN_20220218_022348.mff'},
               {"stim_channel": 'STI 014', "path_in": './sub-s004-ses_07_task-MMN_20220218_022348.mff'}]

bids_path_args = [{'subject': '001', 'run': '01', 'session': '01', "task": "mmn"},
                  {'subject': '002', 'run': '01', 'session': '01', "task": "mmn"}]

bids_paths = ll.bids.convert_dataset_to_bids(egi_import_fct, import_args, bids_path_args, overwrite=True)
```

 Note that, in this case, we used twice the same input file just to demonstrate how this function can be used for multiple recordings. In practice, a user may want to have this information stored in CSV files that can be readily used. For example, if we create such files for the demonstration:

```python
import pandas as pd

pd.DataFrame(import_args).to_csv("import_args.csv", index=False)
pd.DataFrame(bids_path_args).to_csv("bids_path_args.csv", index=False)
```

Now, regardless of how such files have been produced (e.g., from Excel), these can be used directly to process the whole dataset:

```python
import_args = list(pd.read_csv("import_args.csv").T.to_dict().values())
bids_path_args = list(pd.read_csv("bids_path_args.csv").T.to_dict().values())
bids_paths = ll.bids.convert_dataset_to_bids(egi_import_fct, import_args, bids_path_args, overwrite=True)

pipeline.run_dataset(bids_paths)
```

## Running on a cluster or locally with parallelization

TBD

## Manual quality control (QC)

The automated part of the PyLossless pipeline is optimally followed by a manual QC phase. But no sweet! We used the Plotly Dash library to prepared a convenient graphical user interface (GUI) to help you do that! This interface can be lauched using 

```python
$ python ./pylossless/dash/app.py
```

Thanks to it being a web-based GUI, this interface can even be deployed without any local installation, e.g., using a virtualization platform like Google Colab. Here is an [example](https://githubtocolab.com/lina-usc/pylossless/blob/main/notebooks/qc_example.ipynb)!