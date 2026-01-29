from pathlib import Path
from importlib.metadata import version

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


def test_rejection_policy_repr():
    """Test the __repr__ method of RejectionPolicy."""
    rejection_policy = ll.RejectionPolicy(
        ch_flags_to_reject=["noisy"],
        ic_flags_to_reject=["muscle"],
        ic_rejection_threshold=0.5,
        ch_cleaning_mode="drop",
        remove_flagged_ics=False
    )

    repr_str = repr(rejection_policy)

    # Check that repr contains expected fields
    assert "RejectionPolicy:" in repr_str
    assert "ch_flags_to_reject: ['noisy']" in repr_str
    assert "ic_flags_to_reject: ['muscle']" in repr_str
    assert "ic_rejection_threshold: 0.5" in repr_str
    assert "ch_cleaning_mode: drop" in repr_str
    assert "remove_flagged_ics: False" in repr_str


def test_rejection_policy_from_config_file(tmp_path):
    """Test creating RejectionPolicy from config file."""
    # Create a config file
    config_path = tmp_path / "rejection_config.yaml"

    # First create a rejection policy and save it
    rejection_policy = ll.RejectionPolicy(
        ch_flags_to_reject=["noisy", "bridged"],
        ic_flags_to_reject=["muscle", "eog"],
        ic_rejection_threshold=0.4,
        ch_cleaning_mode="interpolate",
        remove_flagged_ics=True
    )
    rejection_policy.save(config_path)

    # Now load it from the config file
    loaded_policy = ll.RejectionPolicy(config_fname=config_path)

    # Verify the loaded policy has the correct values
    assert loaded_policy["ch_flags_to_reject"] == ["noisy", "bridged"]
    assert loaded_policy["ic_flags_to_reject"] == ["muscle", "eog"]
    assert loaded_policy["ic_rejection_threshold"] == 0.4
    assert loaded_policy["ch_cleaning_mode"] == "interpolate"
    assert loaded_policy["remove_flagged_ics"] is True


def test_rejection_policy_legacy_apply(pipeline_fixture):
    """Test legacy apply method (backward compatibility)."""
    # Create a pipeline without operations_log to test legacy path
    pipeline = pipeline_fixture

    # Remove operations_log to force legacy path
    if hasattr(pipeline, 'operations_log'):
        original_log = pipeline.operations_log
        pipeline.operations_log = []

    rejection_policy = ll.RejectionPolicy(
        ch_cleaning_mode="interpolate",
        remove_flagged_ics=True
    )

    # Apply should use legacy method
    cleaned = rejection_policy.apply(pipeline, version_mismatch="ignore")

    # Verify it returns a Raw object
    assert isinstance(cleaned, mne.io.BaseRaw)

    # Restore operations_log
    if 'original_log' in locals():
        pipeline.operations_log = original_log


def test_rejection_policy_notch_and_resample(tmp_path):
    """Test replay with notch_filter and resample operations."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()

    # Add notch filter to config
    config["notch_filter"] = {
        "filter_args": {
            "freqs": [60],
            "notch_widths": [2]
        }
    }

    # Configure resampling
    config["resample"] = {
        "resample_args": {
            "sfreq": 500
        }
    }

    config["ica"] = None  # Skip ICA for speed

    # Create pipeline and manually add operations to test replay
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw_original = raw.copy()
    pipeline.raw = raw.copy()
    pipeline.operations_log = []

    # Manually log filter operation
    pipeline.operations_log.append({
        "operation_id": 0,
        "operation_type": "preprocessing",
        "operation_name": "filter",
        "parameters": {"l_freq": 1.0, "h_freq": 40.0},
        "timestamp": "2026-01-29T00:00:00"
    })

    # Manually log notch filter operation
    # Note: notch_widths omitted to use MNE's default (freqs / 200.0)
    pipeline.operations_log.append({
        "operation_id": 1,
        "operation_type": "preprocessing",
        "operation_name": "notch_filter",
        "parameters": {"freqs": [60]},
        "timestamp": "2026-01-29T00:00:01"
    })

    # Manually log resample operation
    pipeline.operations_log.append({
        "operation_id": 2,
        "operation_type": "preprocessing",
        "operation_name": "resample",
        "parameters": {"sfreq": 500},
        "timestamp": "2026-01-29T00:00:02"
    })

    # Set version
    pipeline.config["version"] = version("pylossless")

    # Test replay
    rejection_policy = ll.RejectionPolicy()
    rejection_policy.remove_flagged_ics = False
    cleaned = rejection_policy.apply(pipeline, version_mismatch="ignore")

    # Verify operations were applied
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.info['sfreq'] == 500  # Check resample was applied


def test_rejection_policy_skip_specific_operations(tmp_path):
    """Test skipping specific preprocessing operations."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None

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

    # Load and apply policy, skipping filter operation
    pipeline_loaded = ll.LosslessPipeline(config=config)
    pipeline_loaded.load_ll_derivative(bids_path)

    rejection_policy = ll.RejectionPolicy()
    rejection_policy.preprocessing_operations_to_skip = ['filter']
    rejection_policy.remove_flagged_ics = False
    cleaned = rejection_policy.apply(pipeline_loaded, version_mismatch="ignore")

    assert isinstance(cleaned, mne.io.BaseRaw)


def test_rejection_policy_uncorrelated_channels(tmp_path):
    """Test handling of uncorrelated channels in artifact flagging."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["ica"] = None

    # Create pipeline with operations log including uncorrelated channels
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw_original = raw.copy()
    pipeline.raw = raw.copy()
    pipeline.operations_log = []

    # Add artifact flag operation with uncorrelated channels
    pipeline.operations_log.append({
        "operation_id": 0,
        "operation_type": "artifact_flag",
        "operation_name": "flag_uncorrelated_channels",
        "parameters": {},
        "flags": {
            "uncorrelated_channels": ["EEG 001", "EEG 002"]
        },
        "timestamp": "2026-01-29T00:00:00"
    })

    pipeline.config["version"] = version("pylossless")

    # Apply rejection policy that includes uncorrelated flags
    rejection_policy = ll.RejectionPolicy(
        ch_flags_to_reject=["uncorrelated"],
        ch_cleaning_mode=None,
        remove_flagged_ics=False
    )
    cleaned = rejection_policy.apply(pipeline, version_mismatch="ignore")

    # Check that uncorrelated channels were marked as bad
    assert "EEG 001" in cleaned.info["bads"]
    assert "EEG 002" in cleaned.info["bads"]


def test_rejection_policy_channels_to_exclude_at_operation(tmp_path):
    """Test _get_channels_to_exclude_at_operation method."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["ica"] = None

    # Create pipeline with operations log
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw_original = raw.copy()
    pipeline.raw = raw.copy()
    pipeline.operations_log = []

    # Add artifact flags before re-referencing
    pipeline.operations_log.append({
        "operation_id": 0,
        "operation_type": "artifact_flag",
        "operation_name": "flag_noisy_channels",
        "parameters": {},
        "flags": {
            "noisy_channels": ["EEG 001"],
            "bridged_channels": ["EEG 002"]
        },
        "timestamp": "2026-01-29T00:00:00"
    })

    # Add re-referencing operation
    pipeline.operations_log.append({
        "operation_id": 1,
        "operation_type": "preprocessing",
        "operation_name": "set_eeg_reference",
        "parameters": {
            "ref_channels": "average"
        },
        "timestamp": "2026-01-29T00:00:01"
    })

    pipeline.config["version"] = version("pylossless")

    # Apply rejection policy
    rejection_policy = ll.RejectionPolicy(
        ch_flags_to_reject=["noisy", "bridged"],
        remove_flagged_ics=False
    )
    cleaned = rejection_policy.apply(pipeline, version_mismatch="ignore")

    # Verify that excluded channels were handled
    assert isinstance(cleaned, mne.io.BaseRaw)


def test_rejection_policy_legacy_apply_with_drop(pipeline_fixture):
    """Test legacy apply method with drop mode."""
    # Create a pipeline without operations_log to test legacy path
    pipeline = pipeline_fixture

    # Remove operations_log to force legacy path
    if hasattr(pipeline, 'operations_log'):
        original_log = pipeline.operations_log
        pipeline.operations_log = []

    rejection_policy = ll.RejectionPolicy(
        ch_cleaning_mode="drop",
        remove_flagged_ics=False
    )

    # Apply should use legacy method
    cleaned = rejection_policy.apply(pipeline, version_mismatch="ignore")

    # Verify it returns a Raw object
    assert isinstance(cleaned, mne.io.BaseRaw)

    # Verify channels were dropped
    flagged_chs = []
    for key in rejection_policy["ch_flags_to_reject"]:
        flagged_chs.extend(pipeline.flags["ch"][key].tolist())

    # Check that dropped channels are not in the cleaned data
    for ch in flagged_chs:
        assert ch not in cleaned.ch_names

    # Restore operations_log
    if 'original_log' in locals():
        pipeline.operations_log = original_log
