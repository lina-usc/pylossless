from pathlib import Path
import shutil
from time import sleep
import pytest

import pylossless as ll

import mne_bids

import openneuro


def load_openneuro_bids():
    config = ll.config.Config()
    config.load_default()
    config['project']['bids_montage'] = ''
    config['project']['analysis_montage'] = 'standard_1020'
    config['project']['set_montage_kwargs']['on_missing'] = 'warn'

    # Shamelessly copied from
    # https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    # pip install openneuro-py

    dataset = 'ds002778'
    subject = 'pd6'

    # Download one subject's data from each dataset
    bids_root = Path('.') / dataset
    # TODO: Delete this directory after test otherwise MNE will think the
    # sample directory is outdated, and will re-download it the next time
    # data_path() is called, which is annoying for users.
    bids_root.mkdir(exist_ok=True)

    openneuro.download(dataset=dataset, target_dir=bids_root,
                       include=[f'sub-{subject}'])

    datatype = 'eeg'
    session = 'off'
    task = 'rest'
    suffix = 'eeg'
    bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task,
                                  suffix=suffix, datatype=datatype,
                                  root=bids_root)

    while not bids_path.fpath.with_suffix('.bdf').exists():
        print(list(bids_path.fpath.glob('*')))
        sleep(1)
    raw = mne_bids.read_raw_bids(bids_path)
    return raw, config, bids_root


# @pytest.mark.xfail
@pytest.mark.parametrize('dataset', ['openneuro'])
def test_pipeline_run(dataset):
    """test running the pipeline."""
    if dataset == 'openneuro':
        raw, config, bids_root = load_openneuro_bids()

    config.save("test_config.yaml")
    pipeline = ll.LosslessPipeline('test_config.yaml')
    not_in_1020 = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
                   'EXG5', 'EXG6', 'EXG7', 'EXG8']
    pipeline.raw = raw.pick('eeg',
                            exclude=not_in_1020).load_data()
    pipeline.run_with_raw(pipeline.raw)
    Path('test_config.yaml').unlink()  # delete config file
    shutil.rmtree(bids_root)
