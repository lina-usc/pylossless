# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Tests for topo_viz.py."""

import mne
from pylossless.dash.mne_visualizer import MNEVisualizer


# chromedriver: https://chromedriver.storage.googleapis.com/
#               index.html?path=114.0.5735.90/
#@pytest.mark.skip(reason="an issue with chromedriver causes failure. Need to debug.")
def test_MNEVisualizer(dash_duo):
    """Test MNEVisualizer."""
    fname = "pylossless/assets/test_data/sub-s01/eeg/sub-s01_task-faceO_eeg.edf"
    raw = mne.io.read_raw_edf(fname)

    mne_viz = MNEVisualizer(app=None, inst=None, mode="standalone",
                            set_callbacks=False)

    dash_duo.start_server(mne_viz.app)
    mne_viz.load_recording(raw)
    assert len(mne_viz.mne_annots.data.durations) == 1605
    assert len(mne_viz.layout.shapes) == 9
