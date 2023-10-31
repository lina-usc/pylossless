import numpy as np

import mne

from mne import make_ad_hoc_cov
from mne.datasets import sample

from mne.simulation import (
    simulate_sparse_stc,
    simulate_raw,
    add_noise,
    add_ecg,
    add_eog,
)


def load_simulated_raw():
    """Load simulated raw data.

    This function returns a modified version of the MNE sample data.

    Returns
    -------
    raw_ : instance of Raw
        Simulated raw data based on the MNE sample data.

    Notes
    -----
    This function is based on the code in
    https://mne.tools/stable/auto_examples/simulation/simulate_raw_data.html
    """
    # LOAD DATA
    data_path = sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    raw_fname = meg_path / "sample_audvis_raw.fif"
    fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    # Load real data as the template
    raw = mne.io.read_raw_fif(raw_fname)
    raw.set_eeg_reference(projection=True)

    # GENERATE DIPOLE TIME SERIES
    n_dipoles = 4  # number of dipoles to create
    epoch_duration = 2.0  # duration of each epoch/event
    rng = np.random.RandomState(0)  # random state (make reproducible)
    np.random.seed(5)

    def data_fun(times, n=0):
        """Generate time-staggered sinusoids at harmonics of 10Hz."""
        n_samp = len(times)
        window = np.zeros(n_samp)
        start, stop = [
            int(ii * float(n_samp) / (2 * n_dipoles)) for ii in (2 * n, 2 * n + 1)
        ]
        window[start:stop] = 1.0
        n += 1
        data = 25e-9 * np.sin(2.0 * np.pi * 10.0 * n * times)
        data *= window
        return data

    times = raw.times[: int(raw.info["sfreq"] * epoch_duration)]
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd["src"]
    stc = simulate_sparse_stc(
        src, n_dipoles=n_dipoles, times=times, data_fun=data_fun, random_state=rng
    )

    # SIMULATE RAW DATA
    raw_sim = simulate_raw(raw.info, [stc] * 10, forward=fwd, verbose=True)
    cov = make_ad_hoc_cov(raw_sim.info)
    add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
    add_ecg(raw_sim, random_state=rng)
    add_eog(raw_sim, random_state=rng)
    raw_sim.pick("eeg")

    # Save Info and Montage for later reuse
    montage = raw_sim.get_montage()
    info = mne.create_info(
        ch_names=raw_sim.ch_names,
        sfreq=raw_sim.info["sfreq"],
        ch_types=raw_sim.get_channel_types(),
    )

    # MAKE A VERY NOISY TIME PERIOD
    raw_selection1 = raw_sim.copy().crop(tmin=0, tmax=2, include_tmax=False)
    raw_selection2 = raw_sim.copy().crop(tmin=2, tmax=3, include_tmax=False)
    raw_selection3 = raw_sim.copy().crop(tmin=3, tmax=19.994505956825666)

    cov_noisy_period = make_ad_hoc_cov(raw_selection2.info, std=dict(eeg=0.000002))
    add_noise(
        raw_selection2, cov_noisy_period, iir_filter=[0.2, -0.2, 0.04], random_state=rng
    )

    raw_selection1.append([raw_selection2, raw_selection3])
    raw_selection1.set_annotations(None)
    raw_sim = raw_selection1

    # MAKE SOME VERY NOISY CHANNELS

    make_these_noisy = ["EEG 001", "EEG 002"]
    raw_noisy_chs = raw_sim.copy().pick(make_these_noisy)
    cov_noisy = make_ad_hoc_cov(raw_noisy_chs.info, std=dict(eeg=0.000002))
    add_noise(raw_noisy_chs, cov_noisy, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

    raw_sim.drop_channels(make_these_noisy)
    raw_noisy_chs.add_channels([raw_sim], force_update_info=True)
    raw_sim = raw_noisy_chs

    # MAKE BRIDGED CHANNELS AND 1 FLAT CHANNEL
    data = raw_sim.get_data()  # ch x times
    data[52, :] = data[53, :]  # duplicate ch 53 and 54

    # MAKE AN UNCORRELATED CH
    min_val = data[28, :].min()
    max_val = data[28, :].min() + 0.0000065
    low_correlated_ch = np.random.uniform(
        low=min_val, high=max_val, size=len(data[28, :])
    )
    data[28] = low_correlated_ch
    # Shuffle it Again. in-place
    # np.random.shuffle(data[23])

    # Make new raw out of data
    raw_sim = mne.io.RawArray(data, info)
    # Re-set the montage
    raw_sim.set_montage(montage)
    return raw_sim
