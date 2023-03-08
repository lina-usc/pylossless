import numpy as np
import mne
from mne import make_ad_hoc_cov
from mne.simulation import (simulate_sparse_stc, simulate_raw, add_noise)

n_dipoles = 4  # number of dipoles to create
epoch_duration = 2.  # duration of each epoch/event
rng = np.random.RandomState(0)  # random state (make reproducible)


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz"""
    n_samp = len(times)
    window = np.zeros(n_samp)
    n = 0  # harmonic number
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 25e-9 * np.sin(2. * np.pi * 10. * n * times)
    data *= window
    return data


def _simulate_raw(raw, fwd_fname,
                  n_dipoles=n_dipoles,
                  epoch_duration=epoch_duration,
                  rng=rng):
    """Simulate a mne raw object with added noise"""
    times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=data_fun, random_state=rng)

    # Simulate Raw Data
    raw_sim = simulate_raw(raw.info, [stc] * 10, forward=fwd, verbose=True)
    raw_sim.pick_types(eeg=True)
    cov = make_ad_hoc_cov(raw_sim.info)
    add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

    make_these_noisy = ['EEG 001', 'EEG 005', 'EEG 009']
    cov_noisy = make_ad_hoc_cov(raw_sim.copy().pick(make_these_noisy).info,
                                std=dict(eeg=.000002))
    add_noise(raw_sim, cov_noisy,
              iir_filter=[0.2, -0.2, 0.04],
              random_state=rng)
    return raw_sim
