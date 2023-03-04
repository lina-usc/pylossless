from pathlib import Path
from time import sleep
import pytest
import shutil

import pylossless as ll

import mne_bids
from mne_bids import write_raw_bids

import openneuro

import mne
from mne.datasets import sample
from mne.datasets.testing import data_path, requires_testing_data

egi_mff_fname = data_path() / 'EGI' / 'test_egi.mff'

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

def test_egi_mff():
    """Test running full pipeline on EGI MFF simple binary files."""
    egi_mff_fname = data_path() / 'EGI' / 'test_egi.mff'
    bids_path = ll.bids.convert_recording_to_bids(mne.io.read_raw_egi,
                        import_kwargs={'input_fname':egi_mff_fname},
                        bids_path_kwargs={'subject':'testegi','task':'test','root':'tmp_test_files'},
                        import_events=False,
                        overwrite=True)

    ll_default_config = ll.config.get_default_config()
    ll_default_config['project']['analysis_montage'] = 'GSN-HydroCel-129'
    ll_default_config['project']['set_montage_kwargs'] = {'match_alias':True}
    ll.config.save_config(ll_default_config, "project_ll_config_test_egi.yaml")

    pipeline = ll.LosslessPipeline('project_ll_config_test_egi.yaml')
    pipeline.run(bids_path, save=False)
    Path('project_ll_config_test_egi.yaml').unlink()
    shutil.rmtree(bids_path.root)


@pytest.mark.parametrize('dataset', ['openneuro'])
def test_pipeline_run(dataset):
    """test running the pipeline."""
    if dataset == 'openneuro':
        raw, ll_default_config = load_openneuro_bids()
    elif dataset == 'egi_mff':
        raw, ll_default_config = load_test_egi_mff()

    ll.config.save_config(ll_default_config, "project_ll_config.yaml")
    pipeline = ll.LosslessPipeline('my_project_ll_config.yaml')
    pipeline.run(raw.pick('eeg', exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']))
    Path('my_project_ll_config.yaml').unlink()  # delete config file we made

# TODO: Add a save-load roundtrip test
#          - Check that FlaggedEpochs indices are preserved save-load roundtrip
