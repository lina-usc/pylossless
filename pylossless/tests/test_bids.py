import pylossless as ll
import mne
import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
def test_find_breaks():
    """Make sure MNE's annotate_break function can run."""
    def edf_import_fct(path_in):
        # read in a file
        raw = mne.io.read_raw_edf(path_in, preload=True)
        print(raw.annotations)
        return raw, np.array([[0, 0, 0]]), {"test": 0, "T0": 1, "T1": 2, "T2": 3}

    testing_path = mne.datasets.testing.data_path()
    fname = testing_path / "EDF" / "test_edf_overlapping_annotations.edf"
    import_args = [{"path_in": fname}]
    bids_path_args = [{'subject': '001', 'run': '01', 'session': '01',
                       "task": "test"}]
    bids_paths = ll.bids.convert_dataset_to_bids(edf_import_fct, import_args,
                                                 bids_path_args, overwrite=True)

    print(bids_paths)

