"""Helper functions and Classes for topographic maps during Lossless QC."""

# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT
from itertools import product
from copy import copy
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd

from mne import create_info
from mne.io import RawArray
from mne.utils.check import _check_sphere
from mne.viz.topomap import _get_pos_outlines
from mne.viz.topomap import _setup_interp, _check_extrapolate

from . import ic_label_cmap
from .css_defaults import CSS, STYLE

# thin lines in the background and numbers below
axis = {'showgrid': False, 'visible': False}
yaxis = copy(axis)
yaxis.update({"scaleanchor": "x", "scaleratio": 1})


class TopoData:  # TODO: Fix/finish doc comments for this class.
    """Handler class for passing Topo Data."""

    def __init__(self, topo_values=()):
        """topo_values: list of dict."""
        self.topo_values = pd.DataFrame(topo_values)

    def add_topomap(self, topomap: dict):
        """topomap: dict."""
        self.topo_values = self.topo_values.append(topomap)

    @property
    def nb_topo(self):
        """topomap: shape."""
        return self.topo_values.shape[0]


class TopoViz:  # TODO: Fix/finish doc comments for this class.
    """Representation of a classic EEG topographic map."""

    def __init__(self, app, montage=None, data=None,  # : TopoData,
                 rows=5, cols=4, width=400, height=600,
                 margin_x=4/5, margin_y=2/5, head_contours_color="black",
                 cmap='RdBu_r', show_sensors=True, show_slider=True,
                 refresh_inputs=None):
        """Initialize instance.

        Parameters
        ----------
        app : instance of Dash.app
            The dash app object to place the plot within.
        montage : mne.channels.DigMontage
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        data : mne.preprocessing.ICA
            The data to use for the topoplots. Can be an instance of
            mne.preprocessing.ICA.
        rows : int
            The number of rows to use for the topoplots. For example, using
            the default values, will show 5 rows of 4 topoplots each. A
            dash dcc.Slider is available to scroll if there are more topoplots
            than can be fit into one row x col view.
        cols : int
            The number of cols to use for the topoplots.
        width : int
            The width of the dcc.graph object holding the topoplots
        height : int
            The height of the dcc.graph object holding the topoplots
        margin_x : float
            Can be a float or for example 4/5.
        margin_y : float
            Can be a float or for example 4/5.
        head_contours_color : str
            The color to use for the topoplot head outline. Must be a string
            of a rgba or hex code that is compatible with plotly's graphing
            library.
        cmap : str
            The color to use for the topoplot heatmap. Must be a string of
            a rgba or hex code that is compatible with plotly's graphing
            library.
        show_sensors : bool
            Whether to show the sensors (as dots) on the topoplot. Defaults
            to True.
        show_slider : bool
            Whether to show the dcc.Slider component that controls which
            topoplots are in view. Defaults to True.
        refresh_inputs : str | iterable
            The id of one or more dash components, that should trigger a
            refresh of the topoplots. For example this can be useful if
            one would like a dcc.dropdown component containing a list of
            file names to refresh the data when selected.
        """
        if not isinstance(refresh_inputs, list):
            refresh_inputs = [refresh_inputs]
        self.refresh_inputs = refresh_inputs
        self.montage = montage
        self.data = data
        self.app = app
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.heatmap_traces = None
        self.colorbar = False
        self.cmap = cmap
        self.titles = None
        self.show_sensors = show_sensors
        self.show_slider = show_slider
        fig = make_subplots(rows=rows, cols=cols,
                            horizontal_spacing=0.01,
                            vertical_spacing=0.01)
        self.graph = dcc.Graph(figure=fig, id='topo-graph',
                               className=CSS['topo-dcc'])
        self.graph_div = html.Div(children=[self.graph],
                                  id='topo-graph-div',
                                  className=CSS['topo-dcc-div'],
                                  style=STYLE['topo-dcc-div'])

        self.margin_x = margin_x
        self.margin_y = margin_y
        self.offset = 0
        self.topo_slider = None
        self.head_contours_color = head_contours_color
        self.info = None
        self.pos = None
        self.contours = None
        self.set_head_pos_contours()

        self.init_slider()
        self.initialize_layout(subplot_titles=self.titles,
                               show_sensors=show_sensors)
        self.set_div()
        self.set_callback()

    def load_recording(self, montage, data):  # TODO: Finish/fix docstring
        """Load recording based on montage and data matrix."""
        self.montage = montage
        self.data = data
        if self.data:
            names = self.data.topo_values.columns.tolist()
            self.info = create_info(names, sfreq=256, ch_types="eeg")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RawArray(np.zeros((len(names), 1)), self.info, copy=None,
                         verbose=False).set_montage(montage)
        self.set_head_pos_contours()
        self.topo_slider.max = self.nb_topo - 1
        self.topo_slider.value = self.nb_topo - 1
        self.initialize_layout()

    # TODO: Finish/fix docstring
    def set_head_pos_contours(self, sphere=None, picks=None):
        """Manually set head position contours."""
        if not self.info:
            return
        sphere = _check_sphere(sphere, self.info)
        self.pos, self.outlines = _get_pos_outlines(self.info, picks, sphere,
                                                    to_sphere=True)

    # TODO: Finish/fix docstring
    def get_head_scatters(self, color="back", show_sensors=True):
        """Build scatter plot from head position data."""
        outlines_scat = [go.Scatter(x=x, y=y, line=dict(color=color),
                                    mode='lines', showlegend=False)
                         for key, (x, y) in self.outlines.items()
                         if 'clip' not in key]
        if show_sensors:
            pos_scat = go.Scatter(x=self.pos.T[0], y=self.pos.T[1],
                                  line=dict(color=color), mode='markers',
                                  marker=dict(color=color,
                                              size=2,
                                              opacity=.5),
                                  showlegend=False)

            return outlines_scat + [pos_scat]

        return outlines_scat

    def get_heatmap_data(self, i, j, ch_type="eeg", res=64,
                         extrapolate='auto'):
        """Get the data to use for the topo plots.

        Parameters
        ----------
        i : int
            The row number
        j : int
            The col number
        ch_type : str
            The data type. defaults to 'eeg'
        res : int
            The dpi resolution for the topoplots. Defaults to 64
        extrapolate : str
            Method to extrapoloate data

        Returns
        -------
        a dict of the data, with keys 'x', 'y', 'z'.
        """
        # TODO : clarify ch_type and res parameters in Docstrong
        # Get the heatmap
        no = i*self.cols + j
        if no + self.offset >= self.data.nb_topo:  # out of range
            return [[]]
        value_dict = dict(self.data.topo_values.iloc[no+self.offset])

        extrapolate = _check_extrapolate(extrapolate, ch_type)
        # find mask limits and setup interpolation
        _, Xi, Yi, interp = _setup_interp(self.pos, res=res,
                                          image_interp="cubic",
                                          extrapolate=extrapolate,
                                          outlines=self.outlines,
                                          border='mean')
        interp.set_values(np.array(list(value_dict.values())))
        Zi = interp.set_locations(Xi, Yi)()

        # Clip to the outer circler
        x0, y0 = self.outlines["clip_origin"]
        x_rad, y_rad = self.outlines["clip_radius"]
        Zi[np.sqrt(((Xi - x0)/x_rad)**2 + ((Yi-y0)/y_rad)**2) > 1] = np.nan

        return {"x": Xi[0], "y": Yi[:, 0], "z": Zi}

    def initialize_layout(self, slider_val=None, subplot_titles=None,
                          show_sensors=True):
        """Initialize the layout for the topoplot dcc.graph component."""
        if not self.data:
            return
        if slider_val is not None:
            self.offset = self.topo_slider.max-slider_val

        ic_names = self.data.topo_values.index
        ic_names = ic_names[self.offset: self.offset+self.rows*self.cols]
        self.graph.figure = make_subplots(rows=self.rows, cols=self.cols,
                                          horizontal_spacing=0.03,
                                          vertical_spacing=0.03,
                                          subplot_titles=ic_names)

        self.heatmap_traces = [[go.Heatmap(showscale=self.colorbar,
                                           colorscale=self.cmap,
                                           **self.get_heatmap_data(i, j))
                                for j in np.arange(self.cols)]
                               for i in np.arange(self.rows)]

        for no, (i, j) in enumerate(product(np.arange(self.rows),
                                            np.arange(self.cols))):
            if no + self.offset >= self.data.nb_topo:  # out of range
                break

            if isinstance(self.head_contours_color, str):
                color = self.head_contours_color
            elif ic_names[no] in self.head_contours_color:
                color = self.head_contours_color[ic_names[no]]
            else:
                color = "black"

            for trace in self.get_head_scatters(color=color,
                                                show_sensors=self.show_sensors
                                                ):
                self.graph.figure.add_trace(trace, row=i+1, col=j+1)
            self.graph.figure.add_trace(self.heatmap_traces[i][j],
                                        row=i+1, col=j+1)

        for i in range(1, self.rows*self.cols+1):
            self.graph.figure['layout'][f'xaxis{i}'].update(axis)
            self.graph.figure['layout'][f'yaxis{i}'].update(yaxis)

        self.graph.figure.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            plot_bgcolor='rgba(0,0,0,0)',  # '#EAEAF2', #'rgba(44,44,44,.5)',
            paper_bgcolor='rgba(0,0,0,0)')  # '#EAEAF2')  #'rgba(44,44,44,.5)')
        self.graph.figure['layout'].update(margin=dict(l=0, r=0, b=0, t=20))

    @property
    def nb_topo(self):
        """The number of topoplots."""
        if self.data:
            return self.data.nb_topo
        return self.rows * self.cols

    def init_slider(self):
        """Initialize the dcc.Slider component for the topoplots."""
        self.topo_slider = dcc.Slider(id='topo-slider',
                                      min=self.rows * self.cols - 1,
                                      max=self.nb_topo - 1,
                                      step=1,
                                      marks=None,
                                      value=self.nb_topo - 1,
                                      included=False,
                                      updatemode='mouseup',
                                      vertical=True,
                                      verticalHeight=400)
        self.topo_slider_div = html.Div(self.topo_slider,
                                        className=CSS['topo-slider-div'],
                                        style={})
        if not self.show_slider:
            self.topo_slider_div.style.update({'display': 'none'})

    def set_div(self):
        """Set the html.Div component for the topoplots."""
        # outer_div includes slider obj
        graph_components = [self.topo_slider_div, self.graph_div]
        self.container_plot = html.Div(children=graph_components,
                                       className=CSS['topo-container'])

    def set_callback(self):
        """Create the callback for the dcc.graph component of the topoplots."""
        args = [Output('topo-graph', 'figure')]
        args += [Input('topo-slider', 'value')]
        if self.refresh_inputs:
            args += self.refresh_inputs

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(slider_val, *args):
            self.initialize_layout(slider_val=slider_val,
                                   subplot_titles=self.titles,
                                   show_sensors=self.show_sensors)
            return self.graph.figure


class TopoVizICA(TopoViz):
    """Representation of a classic ICA topographic map."""

    def __init__(self, app, montage, ica, ic_labels=None, **kwargs):
        """Initialize instance.

        Parameters
        ----------
        app : instance of Dash.app
            The dash app object to place the plot within.
        montage : mne.channels.DigMontage
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        data : mne.preprocessing.ICA
            The data to use for the topoplots. Can be an instance of
            mne.preprocessing.ICA.
        ica : mne.preprocessing.ICA
            The data to use for the topoplots. Can be an instance of
            mne.preprocessing.ICA.
        ic_labels : mapping
            A mapping between the ICA names and their IClabels, which
            can be identified with mne-icalabel soon.
        rows : int
            The number of rows to use for the topoplots. For example, using
            the default values, will show 5 rows of 4 topoplots each. A
            dash dcc.Slider is available to scroll if there are more topoplots
            than can be fit into one row x col view.
        cols : int
            The number of cols to use for the topoplots.
        width : int
            The width of the dcc.graph object holding the topoplots
        height : int
            The height of the dcc.graph object holding the topoplots
        margin_x : float
            Can be a float or for example 4/5.
        margin_y : float
            Can be a float or for example 4/5.
        head_contours_color : str
            The color to use for the topoplot head outline. Must be a string
            of a rgba or hex code that is compatible with plotly's graphing
            library.
        cmap : str
            The color to use for the topoplot heatmap. Must be a string of
            a rgba or hex code that is compatible with plotly's graphing
            library.
        show_sensors : bool
            Whether to show the sensors (as dots) on the topoplot. Defaults
            to True.
        show_slider : bool
            Whether to show the dcc.Slider component that controls which
            topoplots are in view. Defaults to True.
        refresh_inputs : str | iterable
            The id of one or more dash components, that should trigger a
            refresh of the topoplots. For example this can be useful if
            one would like a dcc.dropdown component containing a list of
            file names to refresh the data when selected.
        """
        data = self.init_vars(montage, ica, ic_labels)
        super(TopoVizICA, self).__init__(app, montage, data, **kwargs)

    def init_vars(self, montage, ica, ic_labels):
        """Initialize the montage, ica, and ic_labels data."""
        if not montage or not ica or not ic_labels:
            return None
        if ic_labels:
            self.head_contours_color = {comp: ic_label_cmap[label]
                                        for comp, label
                                        in ic_labels.items()}
        data = TopoData([dict(zip(montage.ch_names, component))
                         for component in ica.get_components().T])
        data.topo_values.index = list(ic_labels.keys())
        return data

    def load_recording(self, montage, ica, ic_labels):
        """Load the object to be plotted."""
        data = self.init_vars(montage, ica, ic_labels)
        super(TopoVizICA, self).load_recording(montage, data)
