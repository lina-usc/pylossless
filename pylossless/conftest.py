"""Pytest fixtures that can be reused across our unit tests."""
# Author: Scott Huberty <seh33@uw.edu>
#
# License: MIT

from pathlib import Path
import shutil

from mne import Annotations

import pylossless as ll
from pylossless.datasets import load_openneuro_bids

import pytest


@pytest.fixture(scope="session")
def pipeline_fixture():
    """Return a namedTuple containing MNE eyetracking raw data and events."""
    raw, config, bids_path = load_openneuro_bids()
    raw.crop(tmin=0, tmax=60)  # take 60 seconds for speed
    annots = Annotations(
        onset=[1, 15], duration=[1, 1], description=["test_annot", "test_annot"]
    )
    raw.set_annotations(annots)

    config["find_breaks"] = {}
    config["find_breaks"]["min_break_duration"] = 9
    config["find_breaks"]["t_start_after_previous"] = 1
    config["find_breaks"]["t_stop_before_next"] = 0
    config.save("test_config.yaml")
    pipeline = ll.LosslessPipeline("test_config.yaml")
    not_in_1020 = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
    pipeline.raw = raw.pick("eeg", exclude=not_in_1020).load_data()
    pipeline.run_with_raw(pipeline.raw)

    Path("test_config.yaml").unlink()  # delete config file
    shutil.rmtree(bids_path.root)
    return pipeline
