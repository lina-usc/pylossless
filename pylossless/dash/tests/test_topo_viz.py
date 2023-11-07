# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Tests for topo_viz.py."""

import pytest

import mne
from dash import html

from pylossless.dash.topo_viz import TopoPlot, GridTopoPlot, TopoData, TopoViz


def get_raw_ica():
    """Get raw and ICA object."""
    data_path = mne.datasets.sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname)
    raw.crop(tmin=100, tmax=130)  # take 30 seconds for speed

    # pick only EEG channels, muscle artifact is basically not picked up by MEG
    # if you have a simultaneous recording, you may want to do ICA on MEG and
    # EEG separately
    raw.pick_types(eeg=True)

    # ICA works best with a highpass filter applied
    raw.load_data()
    raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(max_iter="auto", random_state=97)
    ica.fit(raw)

    return raw, ica


def test_TopoPlot():
    """Test plotting topoplots with plotly."""
    raw, ica = get_raw_ica()
    data = dict(zip(ica.ch_names, ica.get_components()[:, 0]))
    TopoPlot(raw.get_montage(), data, res=200).figure


def test_GridTopoPlot():
    """Test plotting grid of topoplots with plotly."""
    raw, ica = get_raw_ica()

    topo_data = TopoData()
    for comp in ica.get_components().T:
        topo_data.add_topomap(dict(zip(ica.ch_names, comp)))

    offset = 2
    nb_topo = 4
    plot_data = topo_data.topo_values.iloc[::-1].iloc[offset : offset + nb_topo]
    plot_data = list(plot_data.T.to_dict().values())

    GridTopoPlot(
        2,
        2,
        raw.get_montage(),
        plot_data,
        res=200,
        width=300,
        height=300,
        subplots_kwargs=dict(subplot_titles=[1, 2, 3, 4], vertical_spacing=0.05),
    ).figure


# chromedriver: https://chromedriver.storage.googleapis.com/
#               index.html?path=114.0.5735.90/
@pytest.mark.xfail(reason="an issue with chromedriver causes failure. Need to debug.")
def test_TopoViz(dash_duo):
    """Test TopoViz."""
    raw, ica = get_raw_ica()

    topo_data = TopoData()
    for comp in ica.get_components().T:
        topo_data.add_topomap(dict(zip(ica.ch_names, comp)))

    topo_viz = TopoViz(data=topo_data, montage=raw.get_montage(), mode="standalone")

    topo_viz.app.layout.children.append(html.Div(id="nully-wrapper", children=0))
    dash_duo.start_server(topo_viz.app)
    assert dash_duo.find_element("#nully-wrapper").text == "0"
