from pathlib import Path
import shutil
import pytest

import pylossless as ll

import mne

from pylossless.tests.test_datasets import load_openneuro_bids


# @pytest.mark.xfail
@pytest.mark.parametrize(
    "dataset, find_breaks", [("openneuro", True), ("openneuro", False)]
)
def test_pipeline_run(dataset, find_breaks):
    """Test running the pipeline."""
    if dataset == "openneuro":
        raw, config, bids_root = load_openneuro_bids()
    raw.crop(tmin=0, tmax=60)  # take 60 seconds for speed

    if find_breaks:
        config["find_breaks"] = {}
        config["find_breaks"]["min_break_duration"] = 9
        config["find_breaks"]["t_start_after_previous"] = 1
        config["find_breaks"]["t_stop_before_next"] = 0
    config.save("test_config.yaml")
    pipeline = ll.LosslessPipeline("test_config.yaml")
    not_in_1020 = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
    pipeline.raw = raw.pick("eeg", exclude=not_in_1020).load_data()
    pipeline.run_with_raw(pipeline.raw)

    if find_breaks:
        assert "BAD_break" in raw.annotations.description

    Path("test_config.yaml").unlink()  # delete config file
    shutil.rmtree(bids_root)


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
