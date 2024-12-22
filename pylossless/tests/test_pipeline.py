from pathlib import Path
import mne
import mne_bids
import pytest

import pylossless as ll


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


@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
@pytest.mark.filterwarnings("ignore:The provided Epochs instance is not"
                            " filtered between 1 and 100 Hz.")
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
