import pytest

from pathlib import Path

from mne.datasets import sample
import mne_bids

import pylossless as ll
from utils import _simulate_raw

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_raw.fif'
fwd_fname = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'


@pytest.mark.parametrize('raw_fname, fwd_fname',
                         [(raw_fname, fwd_fname)])
def test_flag_outlier_chs(raw_fname, fwd_fname):
    # make BIDS object
    bpath = mne_bids.get_bids_path_from_fname(raw_fname, check=False)
    bpath.suffix = 'sample_audvis_raw'

    # Load real data as the template
    raw = mne_bids.read_raw_bids(bpath)
    raw.set_eeg_reference(projection=True)

    raw_sim = _simulate_raw(raw, fwd_fname)

    # Generate Pipeline Config
    config = ll.config.Config()
    config.load_default()
    config.save("sample_audvis_config.yaml")

    # Run first step of Pipeline
    pipeline = ll.LosslessPipeline('sample_audvis_config.yaml')
    pipeline.raw = raw_sim
    pipeline.flag_outlier_chs()
    # Test
    for noisy_ch in ['EEG 001', 'EEG 005', 'EEG 009']:
        assert noisy_ch in pipeline.flagged_chs['manual']

    # Delete temp config file
    tmp_config_fname = Path(pipeline.config_fname).absolute()
    tmp_config_fname.unlink()  # delete config file
    print('DONE!!')
