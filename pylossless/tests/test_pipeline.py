from pathlib import Path
from time import sleep
import pytest

import pylossless as ll

import mne
import mne_bids

import openneuro


def load_openneuro_bids():
    """Load a BIDS dataset from OpenNeuro."""
    config = ll.config.Config()
    config.load_default()
    config["project"]["bids_montage"] = ""
    config["project"]["analysis_montage"] = "standard_1020"
    config["project"]["set_montage_kwargs"]["on_missing"] = "warn"

    # Shamelessly copied from
    # https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    # pip install openneuro-py

    dataset = "ds002778"
    subject = "pd6"

    # Download one subject's data from each dataset
    bids_root = Path(".") / dataset
    # TODO: Delete this directory after test otherwise MNE will think the
    # sample directory is outdated, and will re-download it the next time
    # data_path() is called, which is annoying for users.
    bids_root.mkdir(exist_ok=True)

    openneuro.download(
        dataset=dataset, target_dir=bids_root, include=[f"sub-{subject}"]
    )

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
        root=bids_root,
    )

    while not bids_path.fpath.with_suffix(".bdf").exists():
        print(list(bids_path.fpath.glob("*")))
        sleep(1)
    raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
    annots = mne.Annotations(
        onset=[1, 15], duration=[1, 1], description=["test_annot", "test_annot"]
    )
    raw.set_annotations(annots)
    return raw, config, bids_root


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
    Path(config_fname).unlink()  # delete config file
