from pathlib import Path
import mne_bids
import pytest

import pylossless as ll

import mne


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


def test_deprecation():
    """Test the config_name property added for deprecation."""
    config = ll.config.Config()
    config.load_default()
    pipeline = ll.LosslessPipeline(config=config)
    # with pytest.raises(DeprecationWarning, match=f"config_fname is deprecated"):
    # DeprecationWarning are currently ignored by pytest given our toml file
    pipeline.config_fname = pipeline.config_fname


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
    pipeline.flags['ch']["bridge"] = ["xx"]
    assert pipeline_fixture.flags['ch'] != pipeline.flags['ch']

    assert pipeline_fixture.flags['epoch'] == pipeline.flags['epoch']
    pipeline.flags['epoch']["bridge"] = ["noisy"]
    assert pipeline_fixture.flags['epoch'] == pipeline.flags['epoch']
