import pytest

from pathlib import Path

import numpy as np

import mne
from mne import make_ad_hoc_cov
from mne.datasets import sample

from mne.simulation import simulate_sparse_stc, simulate_raw, add_noise

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
np.random.seed(5)


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
raw_sim.pick('eeg')

# Save Info and Montage for later re-use
montage = raw_sim.get_montage()
info = mne.create_info(ch_names=raw_sim.ch_names,
                       sfreq=raw_sim.info['sfreq'],
                       ch_types=raw_sim.get_channel_types())


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

make_these_noisy = ['EEG 001', 'EEG 003']
cov_noisy = make_ad_hoc_cov(raw_sim.copy().pick(make_these_noisy).info,
                            std=dict(eeg=.000002))
add_noise(raw_sim, cov_noisy, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

# MAKE LESS NOISY CHANNELS
make_these_noisy = ['EEG 005', 'EEG 007']

raw_selection1 = raw_sim.copy().crop(tmin=0, tmax=8, include_tmax=False)
raw_selection2 = raw_sim.copy().crop(tmin=8, tmax=19.994505956825666)

cov_less_noisy = make_ad_hoc_cov((raw_selection1.copy()
                                                .pick(make_these_noisy)
                                                .info),
                                 std=dict(eeg=.0000008))
add_noise(raw_selection1,
          cov_less_noisy,
          iir_filter=[0.2, -0.2, 0.04],
          random_state=rng)
raw_selection1.append([raw_selection2])
raw_selection1.set_annotations(None)
raw_sim = raw_selection1

# MAKE BRIDGED CHANNELS AND 1 FLAT CHANNEL
data = raw_sim.get_data()  # ch x times
data[52, :] = data[53, :]  # duplicate ch 53 and 54

# Make the last channel random. save for later use
min_val = data[23, :].min()
max_val = data[23, :].min() + .0000065
low_correlated_ch = np.random.uniform(low=min_val,
                                      high=max_val,
                                      size=len(data[23, :]))

# MAKE AN UNCORRELATED CH
data[23] = low_correlated_ch
# Shuffle it Again.
np.random.shuffle(data[23])

# Make new raw out of data
raw_sim = mne.io.RawArray(data, info)
# Re-set the montage
raw_sim.set_montage(montage)

# Make new raw out of data
raw_sim = mne.io.RawArray(data, info)
# Re-set the montage
raw_sim.set_montage(montage)


# LOAD DEFAULT CONFIG
config = ll.config.Config()
config.load_default()
config = ll.config.Config()
config.load_default()

# CUSTOMIZE CONFIG
config['ch_ch_sd']['outliers_kwargs']['k'] = 3
config['ch_ch_sd']['outliers_kwargs']['lower'] = .15
config['ch_ch_sd']['outliers_kwargs']['upper'] = .85

config['epoch_ch_sd']['outliers_kwargs']['k'] = 3
config['epoch_ch_sd']['outliers_kwargs']['lower'] = .15
config['epoch_ch_sd']['outliers_kwargs']['upper'] = .85

config['ch_low_r']['outliers_kwargs']['k'] = 2
config['ch_low_r']['outliers_kwargs']['lower'] = .23
config['ch_low_r']['outliers_kwargs']['upper'] = .85
config['ch_low_r']['flag_crit'] = .25

config['epoch_low_r']['outliers_kwargs']['k'] = 3
config['epoch_low_r']['outliers_kwargs']['lower'] = .15
config['epoch_low_r']['outliers_kwargs']['upper'] = .85

config.save("project_ll_config_face13_egi.yaml")

pipeline = ll.LosslessPipeline('project_ll_config_face13_egi.yaml')
config.save("sample_audvis_config.yaml")

# GENERATE PIPELINE
pipeline = ll.LosslessPipeline('sample_audvis_config.yaml')
pipeline.raw = raw_sim


# TEST
@pytest.mark.parametrize('pipeline',
                         [(pipeline)])
def test_simulated_raw(pipeline):
    pipeline._check_sfreq()
    # This file should have been downsampled
    assert pipeline.raw.info['sfreq'] == 600
    # FIND NOISY EPOCHS
    pipeline.flag_ch_sd_epoch()
    # Epoch 2 was made noisy and should be flagged.
    assert np.array_equal(pipeline.flags['epoch']['ch_sd'], [2])
    epochs = pipeline.get_epochs()
    # only epoch at indice 2 should have been dropped
    assert all(not tup or i == 2 for i, tup in enumerate(epochs.drop_log))

    # RUN FLAG_CH_SD
    pipeline.flag_ch_sd_ch()
    noisy_chs = ['EEG 001', 'EEG 003', 'EEG 005', 'EEG 007']
    assert np.array_equal(pipeline.flags['ch']['ch_sd'], noisy_chs)

    # FIND UNCORRELATED CHS
    data_r_ch = pipeline.flag_ch_low_r()
    # Previously flagged chs should not be in the correlation array
    assert all([name not in data_r_ch.coords['ch']
                for name in pipeline.flags['ch']['ch_sd']])
    # EEG 024 was made random and should be flagged.
    assert ['EEG 024'] in pipeline.flags['ch']['low_r']

    # RUN FLAG_CH_BRIDGE
    data_r_ch = pipeline.flag_ch_low_r()
    pipeline.flag_ch_bridge(data_r_ch)
    # Channels below are duplicates and should be flagged.
    assert 'EEG 053' in pipeline.flags['ch']['bridge']
    assert 'EEG 054' in pipeline.flags['ch']['bridge']

    # Delete temp config file
    tmp_config_fname = Path(pipeline.config_fname).absolute()
    tmp_config_fname.unlink()
