from pathlib import Path
from time import sleep
import pytest

import pylossless as ll

import mne_bids

import openneuro
from mne.datasets import sample

def load_openneuro_bids():
    ll_default_config = ll.config.get_default_config()
    ll_default_config['project']['bids_montage'] = '' 
    ll_default_config['project']['analysis_montage'] = 'standard_1020'
    ll_default_config['project']['set_montage_kwargs']['on_missing'] = 'warn'

    #Shamelessly copied from https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    #pip install openneuro-py

    dataset = 'ds002778'
    subject = 'pd6'

    # Download one subject's data from each dataset
    bids_root = sample.data_path() / dataset
    bids_root.mkdir(exist_ok=True)

    openneuro.download(dataset=dataset, target_dir=bids_root,
                    include=[f'sub-{subject}'])



    datatype = 'eeg'
    session = 'off'
    task = 'rest'
    suffix = 'eeg'
    bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task,
                                suffix=suffix, datatype=datatype, root=bids_root)

    while not bids_path.fpath.with_suffix('.bdf').exists():
        print(list(bids_path.fpath.glob('*')))
        sleep(1)
    raw = mne_bids.read_raw_bids(bids_path)
    return raw, ll_default_config


@pytest.mark.parametrize('dataset', ['openneuro'])
def test_pipeline_run(dataset):
    """test running the pipeline."""
    if dataset == 'openneuro':
        raw, ll_default_config = load_openneuro_bids()
    
    ll.config.save_config(ll_default_config, "my_project_ll_config.yaml")
    pipeline = ll.LosslessPipeline('my_project_ll_config.yaml')
    pipeline.run(raw.pick('eeg', exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']))
    Path('my_project_ll_config.yaml').unlink() # delete config file we made