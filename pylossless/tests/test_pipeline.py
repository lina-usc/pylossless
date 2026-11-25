from pathlib import Path
import mne
import mne_bids
import numpy as np
import pytest
from scipy import signal

import pylossless as ll
from pylossless.utils import _resolve_montage_name


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


@pytest.mark.filterwarnings(
    "ignore:Maximum number of iterations reached:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.filterwarnings("ignore:The provided Epochs instance is not"
                            " filtered between 1 and 100 Hz.")
@pytest.mark.filterwarnings(
    "ignore:The provided ICA instance was fitted with a 'imported_eeglab'"
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
def test_run_ica_with_amica():
    """Test running AMICA as the final ICA method when installed."""
    pytest.importorskip("amica")

    rng = np.random.default_rng(0)
    n_samples = 1000
    time = np.linspace(0, 8, n_samples)
    source_data = np.c_[
        np.sin(2 * time),
        np.sign(np.sin(3 * time)),
        signal.sawtooth(2 * np.pi * time),
        signal.square(5 * time),
    ]
    source_data += 0.2 * rng.standard_normal(source_data.shape)
    source_data /= source_data.std(axis=0)
    mixing = np.array(
        [
            [1, 1, 1, 0.5],
            [0.5, 2, 1, 1.5],
            [1.5, 1, 2, 1],
            [1, 2, 0.5, 2],
        ]
    )
    mixed_data = source_data @ mixing.T

    info = mne.create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=n_samples / 8,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(mixed_data.T, info)
    raw.set_montage(_resolve_montage_name("standard_1020"))

    config = ll.config.Config()
    config.load_default()
    config["epoching"]["epochs_args"]["tmax"] = 1
    config["ica"]["ica_args"]["run2"] = {
        "method": "amica",
        "n_components": 3,
        "n_mixtures": 1,
        "random_state": 0,
        "verbose": 0,
    }

    pipeline = ll.LosslessPipeline(config=config)
    pipeline.raw = raw

    pipeline.run_ica("run2")

    assert pipeline.ica2.n_components_ == 3
    ica_sources = pipeline.ica2.get_sources(pipeline.get_epochs())
    assert ica_sources.get_data().shape[1] == 3
    assert len(pipeline.flags["ic"]) == 3
    assert pipeline.flags["ic"]["ic_type"].notna().all()
    assert pipeline.flags["ic"]["confidence"].notna().all()


@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
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


@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
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


def _make_raw_with_unpositioned_channels():
    """Make EEG data with two channels that have no montage position.

    'FP1' (case mismatch) and 'HEOG' (EOG typed as EEG) have no position.
    """
    rng = np.random.default_rng(0)
    ch_names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "F3", "F4", "P3", "P4",
                "FP1", "HEOG"]
    sfreq = 250
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.standard_normal((len(ch_names), 60 * sfreq)) * 1e-5
    raw = mne.io.RawArray(data, info)
    montage = (
        "colin27_1005"
        if "colin27_1005" in mne.channels.get_builtin_montages()
        else "standard_1005"
    )
    raw.set_montage(montage, on_missing="ignore")
    return raw


@pytest.mark.filterwarnings("ignore:Not setting position")
@pytest.mark.filterwarnings("ignore:There are channels without locations")
@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
@pytest.mark.parametrize("route", ["raw", "bids"])
def test_missing_channel_positions_error(route, tmp_path):
    """Test an informative error names EEG channels without a position."""
    raw = _make_raw_with_unpositioned_channels()
    config = ll.config.Config()
    config.load_default()
    pipeline = ll.LosslessPipeline(config=config)
    match = r"no valid position in the montage.*FP1.*HEOG|no valid position.*HEOG.*FP1"
    if route == "raw":
        with pytest.raises(ValueError, match=match):
            pipeline.run_with_raw(raw)
    else:
        bids_path = mne_bids.BIDSPath(
            subject="001", task="rest", root=tmp_path / "bids", datatype="eeg"
        )
        mne_bids.write_raw_bids(raw, bids_path, format="EDF", overwrite=True,
                                allow_preload=True, verbose=False)
        with pytest.raises(ValueError, match=match) as excinfo:
            pipeline.run(bids_path, save=False)
        # The hint must be actionable for BIDS users too
        assert "electrodes.tsv" in str(excinfo.value)
        assert "analysis_montage" in str(excinfo.value)


def test_no_montage_error():
    """Test an informative error when no montage is set."""
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=100, ch_types="eeg")
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((4, 1000)) * 1e-5, info)
    epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
    with pytest.raises(ValueError, match="No montage is set"):
        ll.pipeline.chan_neighbour_r(epochs, 3, "max")


@pytest.mark.filterwarnings("ignore:Not setting position")
def test_missing_channel_positions_error_names_montage():
    """Test the error names the config montage when one was applied."""
    raw = _make_raw_with_unpositioned_channels()
    raw.set_montage(None)
    montage = (
        "colin27_1005"
        if "colin27_1005" in mne.channels.get_builtin_montages()
        else "standard_1005"
    )
    config = ll.config.Config()
    config.load_default()
    config["project"]["analysis_montage"] = montage
    config["project"]["set_montage_kwargs"] = {"on_missing": "ignore"}
    pipeline = ll.LosslessPipeline(config=config)
    with pytest.raises(ValueError, match=f"in the montage '{montage}'"):
        pipeline.run_with_raw(raw)
