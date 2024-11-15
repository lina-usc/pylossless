import pytest
import shutil


@pytest.mark.filterwarnings("ignore:Converting data files to EDF format")
def test_convert_dataset_to_bids(bids_dataset_fixture):
    """Test the conversion of a recording to a BIDS dataset."""
    shutil.rmtree(bids_dataset_fixture.root)
