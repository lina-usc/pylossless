from pathlib import Path
import json
import mne
import mne_bids
import numpy as np
import pytest

import pylossless as ll

# Apply these filterwarnings to all tests in this module
pytestmark = [
    @pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
    @pytest.mark.filterwarnings("ignore:unique with argument that is not not a Series")
    @pytest.mark.filterwarnings("ignore:No events found or provided")
    @pytest.mark.filterwarnings("ignore:Did not find any events.tsv")
    @pytest.mark.filterwarnings("ignore:Data has a non-integer sampling rate")
    @pytest.mark.filterwarnings("ignore:EDF format requires equal-length data blocks")
]

def test_empty_repr(tmp_path):
    """Test the __repr__ method for a pipeline that hasn't run."""
    config = ll.config.Config()
    config.load_default()
    fpath = tmp_path / "test_config.yaml"
    config.save(fpath)
    pipeline = ll.LosslessPipeline(fpath)
    assert pipeline.__repr__()
    assert pipeline.flags["ch"].__repr__()


def test_pipeline_run(pipeline_fixture):
    """Test running the pipeline."""
    assert "BAD_break" in pipeline_fixture.raw.annotations.description
    assert pipeline_fixture._repr_html_()
    assert pipeline_fixture.flags["ch"].__repr__()


@pytest.mark.filterwarnings("ignore:The provided Epochs instance is not"
                            " filtered between 1 and 100 Hz.")
@pytest.mark.filterwarnings(
    "ignore:FastICA did not converge:sklearn.exceptions.ConvergenceWarning"
    )
def test_pipeline_save(bids_dataset_fixture):
    """Test running the pipeline."""
    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    del config["filtering"]["notch_filter_args"]

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run(bids_dataset_fixture, save=True)

    with pytest.raises(FileExistsError):
        pipeline.save(overwrite=False, format="EDF")
    pipeline.save(overwrite=True, format="EDF")

    # Files are created in a tmp folder so no need
    # to clean up...
    # shutil.rmtree(bids_dataset_fixture.root)


@pytest.mark.parametrize("logging", [True, False])
def test_find_breaks(logging):
    """Make sure MNE's annotate_break function can run."""
    testing_path = mne.datasets.testing.data_path()
    fname = testing_path / "EDF" / "test_edf_overlapping_annotations.edf"
    raw = mne.io.read_raw_edf(fname, preload=True)
    config_fname = "find_breaks_config.yaml"
    config = ll.config.Config()
    config.load_default()
    config["find_breaks"] = {}
    config["find_breaks"]["min_break_duration"] = 15
    config.save(config_fname)
    pipeline = ll.LosslessPipeline(config_fname)
    pipeline.raw = raw
    if logging:
        pipeline.find_breaks(message="Looking for break periods between tasks")
    else:
        pipeline.find_breaks()
        # Now explicitly remove annotations and make sure we avoid MNE's error.
        pipeline.raw.set_annotations(None)
        pipeline.find_breaks()
    Path(config_fname).unlink()  # delete config file


def test_find_outliers():
    """Test the find_outliers method for the case that epochs is None."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.apply_function(lambda x: x * 10, picks="EEG 001") # create an outlier
    config = ll.config.Config().load_default()
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw = raw
    chs_to_leave_out = pipeline.find_outlier_chs()
    assert chs_to_leave_out == ['EEG 001']


def test_find_bads_by_threshold():
    """Test the find bads by threshold function and method."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    # Make a noisy channel
    raw.apply_function(lambda x: x * 3, picks=["EEG 001"])
    epochs = mne.make_fixed_length_epochs(raw, preload=True)

    # First test the function
    with pytest.warns(
        RuntimeWarning, match="The epochs object contains multiple channel types"
        ):
        _ = ll.pipeline.find_bads_by_threshold(epochs)
    epochs.pick("eeg")
    bads = ll.pipeline.find_bads_by_threshold(epochs)
    np.testing.assert_array_equal(bads, ['EEG 001'])

    # Now test the method
    config = ll.config.Config().load_default()
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw = raw
    pipeline.flag_channels_fixed_threshold(threshold=10_000) # too high
    np.testing.assert_array_equal(pipeline.flags["ch"]["volt_std"], [])
    pipeline.flag_channels_fixed_threshold()
    np.testing.assert_array_equal(pipeline.flags["ch"]["volt_std"], ['EEG 001'])


def test_deprecation():
    """Test the config_name property added for deprecation."""
    config = ll.config.Config()
    config.load_default()
    pipeline = ll.LosslessPipeline(config=config)
    # with pytest.raises(DeprecationWarning, match=f"config_fname is deprecated"):
    # DeprecationWarning are currently ignored by pytest given our toml file
    pipeline.config_fname = pipeline.config_fname


def test_multimodality():
    """Test running the pipeline on a multimodal (EEG, MEG) dataset."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.crop(tmin=0, tmax=60)

    config = ll.config.Config()
    config.load_default()
    config["modality"] = ["eeg", "meg"]
    config["ica"] = None
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    assert pipeline.flags["ch"]["noisy"] == ['EEG 007', 'MEG 1032']


def test_load_flags(pipeline_fixture, tmp_path):
    """Test running the pipeline."""
    bids_root = tmp_path / "derivatives" / "pylossless"
    print(bids_root)

    subject = "pd6"
    datatype = "eeg"
    session = "off"
    task = "rest"
    suffix = "eeg"
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root
    )

    pipeline_fixture.save(bids_path,
                          overwrite=False, format="EDF", event_id=None)
    pipeline = ll.LosslessPipeline().load_ll_derivative(bids_path)

    assert pipeline_fixture.flags['ch'] == pipeline.flags['ch']
    pipeline.flags['ch']["bridged"] = ["xx"]
    assert pipeline_fixture.flags['ch'] != pipeline.flags['ch']

    assert pipeline_fixture.flags['epoch'] == pipeline.flags['epoch']
    pipeline.flags['epoch']["bridged"] = ["noisy"]
    assert pipeline_fixture.flags['epoch'] == pipeline.flags['epoch']


def test_original_data_preserved():
    """Test that original data is stored and not modified."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)  # Resample to integer sampling rate for EDF compatibility
    raw.crop(tmax=10)
    original_data = raw.get_data().copy()

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None  # Skip ICA for speed

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    # Original should be stored
    assert pipeline.raw_original is not None

    # Original should be unchanged
    assert np.allclose(pipeline.raw_original.get_data(), original_data)

    # Working copy should be different (preprocessed)
    assert not np.allclose(pipeline.raw.get_data(), original_data)


def test_operations_logged():
    """Test that operations are logged during pipeline execution."""
    fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(eeg=True)
    raw.resample(600)  # Resample to integer sampling rate for EDF compatibility
    raw.crop(tmax=10)

    config = ll.config.Config()
    config.load_default()
    config["filtering"]["filter_args"]["h_freq"] = 40
    config["ica"] = None  # Skip ICA for speed

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    # Should have logged operations
    assert len(pipeline.operations_log) > 0

    # Should have at least one preprocessing operation
    preprocessing_ops = [
        op for op in pipeline.operations_log
        if op["operation_type"] == "preprocessing"
    ]
    assert len(preprocessing_ops) > 0

    # Check operation structure
    for op in pipeline.operations_log:
        assert "operation_id" in op
        assert "operation_type" in op
        assert "operation_name" in op
        assert "timestamp" in op
        assert "parameters" in op


def test_operations_log_saved(tmp_path):
    """Test that operation log is saved to file."""
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

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)
    pipeline.save(bids_path, overwrite=True, format="EDF")

    # Check that operations_log.json exists
    operations_file = bids_root / "operations_log.json"
    assert operations_file.exists()

    # Load and check content
    with open(operations_file) as f:
        data = json.load(f)

    assert "description" in data
    assert "operations" in data
    assert len(data["operations"]) == len(pipeline.operations_log)


def test_operations_log_loaded(tmp_path):
    """Test that operation log is loaded correctly."""
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

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)
    n_operations = len(pipeline.operations_log)

    pipeline.save(bids_path, overwrite=True, format="EDF")

    # Load pipeline
    pipeline_loaded = ll.LosslessPipeline(config=config)
    pipeline_loaded.load_ll_derivative(bids_path)

    # Should have same operations
    assert len(pipeline_loaded.operations_log) == n_operations
