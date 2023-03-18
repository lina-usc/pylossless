import pytest

from pathlib import Path

import numpy as np

import mne
from mne import make_ad_hoc_cov
from mne.datasets import sample

from mne.simulation import (simulate_sparse_stc, simulate_raw, add_noise)

import mne_bids

import pylossless as ll

# LOAD DATA
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_raw.fif'
fwd_fname = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'

# make BIDS object
bpath = mne_bids.get_bids_path_from_fname(raw_fname, check=False)
bpath.suffix = 'sample_audvis_raw'


# Load real data as the template
raw = mne_bids.read_raw_bids(bpath)
raw.set_eeg_reference(projection=True)


# GENERATE DIPOLE TIME SERIES
n_dipoles = 4  # number of dipoles to create
epoch_duration = 2.  # duration of each epoch/event
n = 0  # harmonic number
rng = np.random.RandomState(0)  # random state (make reproducible)


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz"""
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 25e-9 * np.sin(2. * np.pi * 10. * n * times)
    data *= window
    return data


times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']
stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                          data_fun=data_fun, random_state=rng)

# SIMULATE RAW DATA
raw_sim = simulate_raw(raw.info, [stc] * 10, forward=fwd, verbose=True)
raw_sim.pick_types(eeg=True)

# MAKE A VERY NOISY TIME PERIOD
raw_selection1 = raw_sim.copy().crop(tmin=0, tmax=2, include_tmax=False)
raw_selection2 = raw_sim.copy().crop(tmin=2, tmax=3, include_tmax=False)
raw_selection3 = raw_sim.copy().crop(tmin=3, tmax=19.994505956825666)

cov_noisy_period = make_ad_hoc_cov(raw_selection2.info, std=dict(eeg=.000002))
add_noise(raw_selection2,
          cov_noisy_period,
          iir_filter=[0.2, -0.2, 0.04],
          random_state=rng)

raw_selection1.append([raw_selection2, raw_selection3])
raw_selection1.set_annotations(None)
raw_sim = raw_selection1

# MAKE SOME VERY NOISY CHANNELS
cov = make_ad_hoc_cov(raw_sim.info)
add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

make_these_noisy = ['EEG 001', 'EEG 005', 'EEG 009']
cov_noisy = make_ad_hoc_cov(raw_sim.copy().pick(make_these_noisy).info,
                            std=dict(eeg=.000002))
add_noise(raw_sim, cov_noisy, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

# MAKE LESS NOISY CHANNELS
make_these_noisy = ['EEG 015', 'EEG 016']

raw_selection1 = raw_sim.copy().crop(tmin=0, tmax=8, include_tmax=False)
raw_selection2 = raw_sim.copy().crop(tmin=8, tmax=19.994505956825666)

cov_less_noisy = make_ad_hoc_cov(raw_selection1.copy().pick(make_these_noisy).info,
                                 std=dict(eeg=.0000008))
add_noise(raw_selection1,
          cov_less_noisy,
          iir_filter=[0.2, -0.2, 0.04],
          random_state=rng)
raw_selection1.append([raw_selection2])
raw_selection1.set_annotations(None)
raw_sim = raw_selection1

# LOAD DEFAULT CONFIG
config = ll.config.Config()
config.load_default()
config.save("sample_audvis_config.yaml")

# GENERATE PIPELINE
pipeline = ll.LosslessPipeline('sample_audvis_config.yaml')
pipeline.raw = raw_sim


@pytest.mark.parametrize('pipeline',
                         [(pipeline)])
def test_simulated_raw(pipeline):
    # RUN FLAG OUTLIER CHS
    pipeline.flag_outlier_chs()
    assert np.array_equal(pipeline.flagged_chs['outliers'],
                          ['EEG 001', 'EEG 005', 'EEG 009'])
    assert np.array_equal(pipeline.flagged_chs['manual'],
                          ['EEG 001', 'EEG 005', 'EEG 009'])

    pipeline.flag_ch_sd_epoch()
    assert np.array_equal(pipeline.flagged_epochs['ch_sd'],
                          [2])

    # RUN FLAG_CH_SD
    pipeline.flag_ch_sd_ch()
    assert np.array_equal(pipeline.flagged_chs['ch_sd'],
                          ['EEG 015', 'EEG 016'])
    assert np.array_equal(pipeline.flagged_chs['manual'],
                          ['EEG 001', 'EEG 005', 'EEG 009',
                           'EEG 015', 'EEG 016'])

    # Delete temp config file
    tmp_config_fname = Path(pipeline.config_fname).absolute()
    tmp_config_fname.unlink()
