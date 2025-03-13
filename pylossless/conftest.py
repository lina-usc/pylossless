"""Pytest fixtures that can be reused across our unit tests."""
# Author: Scott Huberty <seh33@uw.edu>
#         Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: MIT

from pathlib import Path
import shutil

import mne
import numpy as np
from mne import Annotations

import pylossless as ll
from pylossless.datasets import load_openneuro_bids

import pytest


@pytest.fixture(scope="session")
def pipeline_fixture():
    """Return a LosslessPipeline object."""
    raw, config, bids_path = load_openneuro_bids()
    # raw.crop(tmin=0, tmax=60)  # Too short for ICA to converge in some tests.
    annots = Annotations(
        onset=[1, 15], duration=[1, 1], description=["test_annot", "test_annot"]
    )
    raw.set_annotations(annots)

    config["find_breaks"] = {}
    config["find_breaks"]["min_break_duration"] = 9
    config["find_breaks"]["t_start_after_previous"] = 1
    config["find_breaks"]["t_stop_before_next"] = 0
    config["flag_channels_fixed_threshold"] = {"threshold": 10_000}
    config["ica"]["ica_args"]["run1"]["max_iter"] = 5000

    # Testing when passing the config object directly...
    pipeline = ll.LosslessPipeline("test_config.yaml", config)
    pipeline = ll.LosslessPipeline(config=config)
    config.save("test_config.yaml")

    # Testing when passing a string...
    pipeline = ll.LosslessPipeline("test_config.yaml")

    # Testing when passing a Path...
    pipeline = ll.LosslessPipeline(Path("test_config.yaml"))

    not_in_1020 = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
    pipeline.raw = raw.pick("eeg", exclude=not_in_1020).load_data()
    pipeline.run_with_raw(pipeline.raw)

    Path("test_config.yaml").unlink()  # delete config file
    shutil.rmtree(bids_path.root)
    return pipeline


@pytest.fixture(scope="session")
@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
def bids_dataset_fixture(tmpdir_factory):
    """Return a BIDS path for a test recording."""
    def edf_import_fct(path_in):
        # read in a file
        raw = mne.io.read_raw_edf(path_in, preload=True)
        match_alias = {ch_name: ch_name.strip(".") for ch_name in raw.ch_names}
        raw.set_montage("standard_1005", match_alias=match_alias, match_case=False)
        return raw, np.array([[0, 0, 0]]), {"test": 0, "T0": 1, "T1": 2, "T2": 3}

    tmp_path = tmpdir_factory.mktemp('bids_dataset')
    testing_path = mne.datasets.testing.data_path()
    fname = testing_path / "EDF" / "test_edf_overlapping_annotations.edf"
    import_args = [{"path_in": fname}]
    bids_path_args = [{'subject': '001', 'run': '01', 'session': '01',
                       "task": "test"}]
    bids_path = ll.bids.convert_dataset_to_bids(
        edf_import_fct,
        import_args,
        bids_path_args,
        bids_root=tmp_path,
        overwrite=True
    )[0]
    return bids_path
