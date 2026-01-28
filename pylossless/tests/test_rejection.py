from pathlib import Path

from numpy.testing import assert_array_equal
import mne
import mne_bids

import pytest

import pylossless as ll

# Apply these filterwarnings to all tests in this module
pytestmark = [
    pytest.mark.filterwarnings("ignore:Converting data files to EDF format"),
    pytest.mark.filterwarnings("ignore:unique with argument that is not not a Series"),
    pytest.mark.filterwarnings("ignore:No events found or provided"),
    pytest.mark.filterwarnings("ignore:Did not find any events.tsv"),
    pytest.mark.filterwarnings("ignore:Data has a non-integer sampling rate"),
    pytest.mark.filterwarnings("ignore:EDF format requires equal-length data blocks"),
]


@pytest.mark.parametrize("clean_ch_mode", [None, "drop", "interpolate"])
def test_rejection_policy(clean_ch_mode, pipeline_fixture):
    """Test the rejection policy."""
    rejection_config_path = Path("test_rejection_config.yaml")
    rejection_config = ll.RejectionPolicy(ch_cleaning_mode=clean_ch_mode)
    rejection_config.save(rejection_config_path)
    want_flags = ["noisy", "uncorrelated", "bridged"]
    assert rejection_config["ch_flags_to_reject"] == want_flags

    pipeline_fixture.config["version"] = "-1"
    with pytest.raises(RuntimeError, match="The output of the pipeline was"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="raise")
    with pytest.raises(RuntimeWarning, match="The output of the pipeline was"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="warning")
    with pytest.raises(ValueError, match="version_mismatch can take values"):
        raw, ica = rejection_config.apply(pipeline_fixture,
                                          version_mismatch="sdfdf")
    raw, ica = rejection_config.apply(pipeline_fixture, return_ica=True,
                                      version_mismatch="ignore")

    flagged_chs = []
    for key in rejection_config["ch_flags_to_reject"]:
        flagged_chs.extend(pipeline_fixture.flags["ch"][key].tolist())
    assert flagged_chs == ["PO3", "Oz", "O2"]
    if clean_ch_mode is None:
        assert_array_equal(flagged_chs, raw.info["bads"])
    elif clean_ch_mode == "drop":
        assert len(list(set(flagged_chs) - set(raw.ch_names))) == 3
    elif clean_ch_mode == "interpolate":
        # interpolate_bads will drop ch_names from raw.info["bads"]
        # so to make sure that the channel was interpolated, lets check it
        assert len(list(set(flagged_chs) - set(raw.info["bads"]))) == 3

    df = pipeline_fixture.flags["ic"]
    assert not df.loc[ica.exclude]["ic_type"].str.contains("brain").any()
    assert not df.loc[ica.exclude]["ic_type"].str.contains("other").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("muscle").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("eog").any()
    assert df.loc[ica.exclude]["ic_type"].str.contains("line_noise").any()
    threshold = rejection_config["ic_rejection_threshold"]
    assert (df.loc[ica.exclude]["confidence"] > threshold).all()
    rejection_config_path.unlink()


def test_rejection_policy_replay(tmp_path):
    """Test operation replay via RejectionPolicy with new lossless format."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)  # Resample to integer sampling rate for EDF compatibility
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None  # Skip ICA for speed

    # Create BIDS structure for saving
    subject = "test"
    datatype = "eeg"
    task = "test"
    suffix = "eeg"
    bids_root = tmp_path / "derivatives" / "pylossless"

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root
    )

    # Run and save pipeline
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)
    pipeline.save(bids_path, overwrite=True, format="EDF")

    # Load and apply policy
    pipeline_loaded = ll.LosslessPipeline(config=config)
    pipeline_loaded.load_ll_derivative(bids_path)

    rejection_policy = ll.RejectionPolicy()
    rejection_policy.remove_flagged_ics = False  # No ICA in this test
    cleaned = rejection_policy.apply(pipeline_loaded, version_mismatch="ignore")

    # Should return mne.io.BaseRaw object (or any subclass like RawEDF, RawFIF, etc.)
    assert isinstance(cleaned, mne.io.BaseRaw)

    # Should have operations log
    assert hasattr(pipeline_loaded, 'operations_log')
    assert len(pipeline_loaded.operations_log) > 0


def test_rejection_policy_skip_preprocessing(tmp_path):
    """Test skipping preprocessing operations in replay."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)  # Resample to integer sampling rate for EDF compatibility
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None  # Skip ICA for speed

    # Create BIDS structure for saving
    subject = "test"
    datatype = "eeg"
    task = "test"
    suffix = "eeg"
    bids_root = tmp_path / "derivatives" / "pylossless"

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root
    )

    # Run and save pipeline
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)
    pipeline.save(bids_path, overwrite=True, format="EDF")

    # Load and apply policy without preprocessing
    pipeline_loaded = ll.LosslessPipeline(config=config)
    pipeline_loaded.load_ll_derivative(bids_path)

    rejection_policy = ll.RejectionPolicy()
    rejection_policy.apply_preprocessing = False
    rejection_policy.remove_flagged_ics = False
    cleaned = rejection_policy.apply(pipeline_loaded, version_mismatch="ignore")

    # Should return mne.io.BaseRaw object (or any subclass like RawEDF, RawFIF, etc.)
    assert isinstance(cleaned, mne.io.BaseRaw)


def test_rejection_policy_param_override(tmp_path):
    """Test overriding operation parameters during replay."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)  # Resample to integer sampling rate for EDF compatibility
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None  # Skip ICA for speed

    # Create BIDS structure for saving
    subject = "test"
    datatype = "eeg"
    task = "test"
    suffix = "eeg"
    bids_root = tmp_path / "derivatives" / "pylossless"

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root
    )

    # Run and save pipeline
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)
    pipeline.save(bids_path, overwrite=True, format="EDF")

    # Load and apply policy with custom filter parameters
    pipeline_loaded = ll.LosslessPipeline(config=config)
    pipeline_loaded.load_ll_derivative(bids_path)

    rejection_policy = ll.RejectionPolicy()
    rejection_policy.operation_param_overrides = {
        'filter': {'l_freq': 0.5, 'h_freq': 30.0}
    }
    rejection_policy.remove_flagged_ics = False
    cleaned = rejection_policy.apply(pipeline_loaded, version_mismatch="ignore")

    # Should return mne.io.BaseRaw object (or any subclass like RawEDF, RawFIF, etc.)
    assert isinstance(cleaned, mne.io.BaseRaw)
