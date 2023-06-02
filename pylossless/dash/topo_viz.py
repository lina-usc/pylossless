"""Helper functions and Classes for topographic maps during Lossless QC."""

# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT
from copy import copy
import warnings
from collections import OrderedDict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash

import mne
import numpy as np
import pandas as pd

from mne import create_info
from mne.io import RawArray
from mne.utils.check import _check_sphere
from mne.viz.topomap import _get_pos_outlines
from mne.viz.topomap import _setup_interp, _check_extrapolate

from .css_defaults import CSS, STYLE
from . import ic_label_cmap

# thin lines in the background and numbers below
axis = {'showgrid': False, 'visible': False}
yaxis = copy(axis)
yaxis.update({"scaleanchor": "x", "scaleratio": 1})


class TopoPlot:  # TODO: Fix/finish doc comments for this class.
    """Representation of a classic EEG topographic map as a plotly figure."""

    def __init__(self, montage="standard_1020", data=None, figure=None,
                 color="black", row=None, col=None, res=64, width=None,
                 height=None, cmap='RdBu_r', show_sensors=True,
                 colorbar=False):
        """Initialize instance.

        Parameters
        ----------
        montage : mne.channels.DigMontage | str
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        data : OrderedDict | None
            The data to use for the topoplots.
        figure : plotly.graph_objects.Figure | None
            Figure to use (if not None) for plotting.
        color : str
            The color to use for the topoplot head outline. Must be a string
            of a rgba or hex code that is compatible with plotly's graphing
            library.
        row : int | None
            Row number of the topoplot, if embedded in a grid.
        col : int | None
            Column number for the topoplot, if embedded in a grid.
        res : int
            Resolution (res X res) for the heatmap.
        width : int
            The width of the dcc.graph object holding the topoplots
        height : int
            The height of the dcc.graph object holding the topoplots
        cmap : str
            The color to use for the topoplot heatmap. Must be a string of
            a rgba or hex code that is compatible with plotly's graphing
            library.
        show_sensors : bool
            Whether to show the sensors (as dots) on the topoplot. Defaults
            to True.
        colorbar : bool
            Whether to show the colorbar.
        """
        self.heatmap_traces = None
        self.colorbar = colorbar
        self.cmap = cmap
        self.title = None
        self.show_sensors = show_sensors
        self.info = None
        self.pos = None
        self.contours = None
        self.__data = None
        self.pos = None
        self.outlines = None
        self.color = color
        self.info = None
        self.col = col
        self.row = row
        self.res = res
        self.width = width
        self.height = height

        if isinstance(montage, str):
            self.montage = mne.channels.make_standard_montage(montage)
        else:
            self.montage = montage

        if figure is None:
            self.figure = go.Figure()
        else:
            self.figure = figure

        if data is None:
            return

        self.data = data
        self.plot_topo()

    @property
    def data(self):
        """Get accessor for the object containing the topomap data.

        Returns
        -------
        An OrderedDict with the channel names as keys, and the values
        to plot as values.
        """
        return self.__data

    @data.setter
    def data(self, data):
        """Setter accessor for the data of the topomap.

        Parameters
        ----------
        data : dict
            Dictionary of channel name (key) and corresponding values
            to be plotted.
        """
        self.set_data(data)

    def set_data(self, data):
        """Set the data used for plotting.

        Parameters
        ----------
        data : dict
            Dictionary of channel name (key) and corresponding values
            to be plotted.
        """
        if data is None:
            return
        self.__data = OrderedDict(data)
        names = list(self.__data.keys())
        self.info = create_info(names, sfreq=256, ch_types="eeg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RawArray(np.zeros((len(names), 1)), self.info, copy=None,
                     verbose=False).set_montage(self.montage)
        self.set_head_pos_contours()

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
                         if 'clip' not in key and "mask" not in key]
        if show_sensors:
            pos_scat = go.Scatter(x=self.pos.T[0], y=self.pos.T[1],
                                  line=dict(color=color), mode='markers',
                                  marker=dict(color=color,
                                              size=2,
                                              opacity=.5),
                                  showlegend=False)

            return outlines_scat + [pos_scat]

        return outlines_scat

    def get_heatmap_data(self, ch_type="eeg", extrapolate='auto'):
        """Get the data to use for the topo plots.

        Parameters
        ----------
        data : dict
            Dictionary with channel names as key and float as values.
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
        extrapolate = _check_extrapolate(extrapolate, ch_type)
        # find mask limits and setup interpolation
        _, Xi, Yi, interp = _setup_interp(self.pos, res=self.res,
                                          image_interp="cubic",
                                          extrapolate=extrapolate,
                                          outlines=self.outlines,
                                          border='mean')
        interp.set_values(np.array(list(self.__data.values())))
        Zi = interp.set_locations(Xi, Yi)()

        # Clip to the outer circler
        x0, y0 = self.outlines["clip_origin"]
        x_rad, y_rad = self.outlines["clip_radius"]
        Zi[np.sqrt(((Xi - x0)/x_rad)**2 + ((Yi-y0)/y_rad)**2) > 1] = np.nan

        return {"x": Xi[0], "y": Yi[:, 0], "z": Zi}

    def _update_axes(self):
        self.figure.update_xaxes({'showgrid': False, 'visible': False},
                                 row=self.row, col=self.col)

        scale_anchor = list(self.figure.select_yaxes(row=self.row,
                                                     col=self.col))
        scale_anchor = scale_anchor[0]["anchor"]
        if not scale_anchor:
            scale_anchor = "x"
        self.figure.update_yaxes({'showgrid': False, 'visible': False,
                                  "scaleanchor": scale_anchor,
                                  "scaleratio": 1},
                                 row=self.row, col=self.col)

        self.figure.update_layout(
                    autosize=False,
                    width=self.width,
                    height=self.height,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, b=0, t=20))

    def plot_topo(self, **kwargs):
        """Plot the topomap.

        Parameters
        ----------
        **kwargs
            Arguments pass to plotly.graph_objects.Heatmap.

        Returns
        -------
            A plotly.graph_objects.Figure object.
        """
        if self.__data is None:
            return

        heatmap_trace = go.Heatmap(showscale=self.colorbar,
                                   colorscale=self.cmap,
                                   **self.get_heatmap_data(**kwargs))

        for trace in self.get_head_scatters(color=self.color):
            self.figure.add_trace(trace, row=self.row, col=self.col)
        self.figure.add_trace(heatmap_trace, row=self.row, col=self.col)

        self._update_axes()

        return self.figure


def __check_shape__(rows, cols, data, fill=None):
    if not isinstance(data, (list, tuple, np.ndarray)):
        return np.array([[data]*cols]*rows)

    data = np.array(data)
    if data.shape == (rows, cols):
        return data

    if len(data.ravel()) < rows*cols:
        data = np.concatenate((data.ravel(),
                               [fill]*(rows*cols-len(data.ravel()))))
    return data.reshape((rows, cols))


class GridTopoPlot:
    """Representation of grid of topomaps as a plotly figure."""

    def __init__(self, rows=1, cols=1, montage="standard_1020",
                 data=None, figure=None, color="black",
                 subplots_kwargs=None, **kwargs):
        """Initialize instance.

        Parameters
        ----------
        rows : int
            The number of rows to use for the topoplots. For example, using
            the default values, will show 5 rows of 4 topoplots each. A
            dash dcc.Slider is available to scroll if there are more topoplots
            than can be fit into one row x col view.
        cols : int
            The number of cols to use for the topoplots.
        montage : mne.channels.DigMontage
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        data : mne.preprocessing.ICA | None
            The data to use for the topoplots. Can be an instance of
            mne.preprocessing.ICA.
        figure : plotly.graph_objects.Figure | None
            Figure to use (if not None) for plotting.
        color : str
            The color to use for the topoplot head outline. Must be a string
            of a rgba or hex code that is compatible with plotly's graphing
            library.
        subplots_kwargs : dict
            Arguments to be passed to plotly.subplots.make_subplots.
        **kwargs:
            Additional arguments to be passed to TopoPlot.
        """
        montage = __check_shape__(rows, cols, montage)
        color = __check_shape__(rows, cols, color)

        subplots_kwargs_ = dict(horizontal_spacing=0.03,
                                vertical_spacing=0.03)
        if subplots_kwargs:
            subplots_kwargs_.update(subplots_kwargs)
        self.rows = rows
        self.cols = cols
        self.color = color
        self.figure = None

        if data is None:
            self.__data = None
            return
        self.__data = __check_shape__(rows, cols, data)

        if figure is None:
            self.figure = make_subplots(rows=rows, cols=cols,
                                        **subplots_kwargs_)
        else:
            self.figure = figure

        self.topos = np.array([[TopoPlot(montage=m, data=d,
                                         figure=self.figure, col=col+1,
                                         row=row+1, color=color, **kwargs)
                                for col, (m, d, color)
                                in enumerate(zip(montage_row, data_row,
                                                 color_row))]
                               for row, (montage_row, data_row, color_row)
                               in enumerate(zip(montage, self.__data,
                                                self.color))])

    @property
    def nb_topo(self):
        """Return the number of topoplots."""
        return self.rows*self.cols


class TopoData:  # TODO: Fix/finish doc comments for this class.
    """Handler class for passing Topo Data."""

    def __init__(self, topo_values=()):
        """topo_values: list of dict."""
        self.topo_values = pd.DataFrame(topo_values)

    def add_topomap(self, topomap: dict, title=None):
        """topomap: dict."""
        if not title:
            title = str(len(self.topo_values))
        self.topo_values = pd.concat([self.topo_values,
                                      pd.DataFrame(topomap, index=[title])])

    @property
    def nb_topo(self):
        """topomap: shape."""
        return self.topo_values.shape[0]


class TopoViz:  # TODO: Fix/finish doc comments for this class.
    """Representation of a classic EEG topographic map."""

    def __init__(self, app=None, montage=None, data=None, rows=5, cols=4,
                 width=400, height=600, margin_x=4/5, margin_y=2/5, res=64,
                 head_contours_color="black",
                 cmap='RdBu_r', show_sensors=True, mode=None,
                 show_slider=True, refresh_inputs=None):
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
        res: int
            Resolution (res X res) of the heatmaps generated for the topomaps.
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
        mode : str
            Can take the value "standalone_jupyter" for an app within a
            Jupyter Notebook, "standalone" for a typical Dash app, or
            "embedded" for using an pre-existing app object.
        show_slider : bool
            Whether to show the dcc.Slider component that controls which
            topoplots are in view. Defaults to True.
        refresh_inputs : str | iterable
            The id of one or more dash components, that should trigger a
            refresh of the topoplots. For example this can be useful if
            one would like a dcc.dropdown component containing a list of
            file names to refresh the data when selected.
        """
        if refresh_inputs:
            if not isinstance(refresh_inputs, list):
                refresh_inputs = [refresh_inputs]
        else:
            refresh_inputs = []

        self.refresh_inputs = refresh_inputs
        self.app = app
        self.show_slider = show_slider
        self.data = None
        self.montage = montage
        self.offset = 0
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.res = res
        self.colorbar = False
        self.cmap = cmap
        self.titles = None
        self.show_sensors = show_sensors
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.head_contours_color = None
        self.container_plot = None

        if app is None:
            stylesheets = [dbc.themes.SLATE]
            if mode == "standalone_jupyter":
                from jupyter_dash import JupyterDash
                self.app = JupyterDash("TopoViz",
                                       external_stylesheets=stylesheets)
                self.mode = mode
            else:
                self.app = dash.Dash("TopoViz",
                                     external_stylesheets=stylesheets)
                self.mode = "standalone"
            self.app.layout = html.Div([])
        else:
            self.app = app
            self.mode = "embedded"

        self.graph = dcc.Graph(figure=None, id='topo-graph',
                               className=CSS['topo-dcc'])
        self.graph_div = html.Div(children=[self.graph],
                                  id='topo-graph-div',
                                  className=CSS['topo-dcc-div'],
                                  style=STYLE['topo-dcc-div'])

        self._init_slider()
        self.set_data(montage, data, head_contours_color)
        self._set_div()
        self.figure = None

        self._set_callback()

        if "standalone" in self.mode:
            self.app.layout.children.append(self.container_plot)

    @property
    def figure(self):
        """Return the go.Graph.figure object."""
        return self.graph.figure

    @figure.setter
    def figure(self, fig):
        """Set the value of the plotly figure object.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure | None
            Figure object to be set or None if no figure should be plotted.
        """
        if fig:
            self.graph.figure = fig
        else:
            self.graph.figure = {}

    def set_data(self, montage=None, data=None, head_contours_color="black"):
        """Set the data used for plotting.

        Parameters
        ----------
        montage : mne.channels.DigMontage
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        data : TopoData
            Data to be plotted.
        head_contour_color : str
            The color to use for the topoplot head outline. Must be a string
            of a rgba or hex code that is compatible with plotly's graphing
            library.
        """
        if data is None:
            return
        self.data = data
        if montage is not None:
            self.montage = montage
        if isinstance(head_contours_color, str):
            head_contours_color = {title: head_contours_color
                                   for title in self.data.topo_values.index}
        if head_contours_color:
            self.head_contours_color = head_contours_color

        self.topo_slider.max = self.nb_topo - 1
        self.topo_slider.value = self.nb_topo - 1
        self.initialize_layout()

    def initialize_layout(self, slider_val=None, show_sensors=True):
        """Initialize the layout for the topoplot dcc.graph component."""
        if self.data is None:
            self.figure = None
            return

        if slider_val is not None:
            """
            For a total of N topomaps and a grid of M topomaps being displayed,
            the values for the slide varies from [M-1, N-1] with an initial
            value at N-1.
            """
            self.offset = slider_val - self.nb_sel_topo + 1

        titles = self.data.topo_values.index

        last_sel_topo = self.offset+self.nb_sel_topo
        titles = titles[::-1][self.offset:last_sel_topo][::-1]
        colors = [self.head_contours_color[title] for title in titles]

        # The indexing with ch_names is to ensure the order
        # of the channels are compatible between plot_data and the montage
        ch_names = [ch_name for ch_name in self.montage.ch_names
                    if ch_name in self.data.topo_values.columns]
        plot_data = self.data.topo_values.loc[titles, ch_names]
        plot_data = list(plot_data.T.to_dict().values())

        if len(plot_data) < self.nb_sel_topo:
            nb_missing_topo = self.nb_sel_topo-len(plot_data)
            plot_data = np.concatenate((plot_data,
                                        [None]*nb_missing_topo))

        self.figure = GridTopoPlot(rows=self.rows, cols=self.cols,
                                   montage=self.montage, data=plot_data,
                                   color=colors,
                                   res=self.res,
                                   height=self.height,
                                   width=self.width,
                                   show_sensors=show_sensors,
                                   subplots_kwargs=dict(
                                        horizontal_spacing=0.03,
                                        vertical_spacing=0.03,
                                        subplot_titles=titles,
                                        )
                                   ).figure

    @property
    def nb_sel_topo(self):
        """Return the number of visible topoplots."""
        return self.rows * self.cols

    @property
    def nb_topo(self):
        """Return the total number of topoplots.

        Returns
        -------
           The total thumber of topomaps, which may be a faction of
           the number of visible topomaps at a givent time.
        """
        if self.data:
            return self.data.nb_topo
        return 0

    def _init_slider(self):
        """Initialize the dcc.Slider component for the topoplots."""
        self.topo_slider = dcc.Slider(id='topo-slider',
                                      min=self.nb_sel_topo - 1,
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

    def _set_div(self):
        """Set the html.Div component for the topoplots."""
        # outer_div includes slider obj
        graph_components = [self.topo_slider_div, self.graph_div]
        self.container_plot = html.Div(children=graph_components,
                                       id="ica-topo-div",
                                       className=CSS['topo-container'],
                                       style={'display': 'none'})

    def _set_callback(self):
        """Create the callback for the dcc.graph component of the topoplots."""
        args = [Output('topo-graph', 'figure')]
        args += [Input('topo-slider', 'value')]
        if self.refresh_inputs:
            args += self.refresh_inputs

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(slider_val, *args):
            self.initialize_layout(slider_val=slider_val,
                                   show_sensors=self.show_sensors)
            if self.figure:
                return self.figure
            return dash.no_update

        @self.app.callback(Output('ica-topo-div', 'style'),
                           Input('topo-graph', 'figure'),
                           )
        def show_figure(figure):
            if figure:
                return {'display': 'block'}
            return {'display': 'none'}


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
        if not montage or not ica:
            return None

        data = TopoData([dict(zip(montage.ch_names, component))
                         for component in ica.get_components().T])
        if ic_labels:
            self.head_contours_color = {comp: ic_label_cmap[label]
                                        for comp, label
                                        in ic_labels.items()}
            data.topo_values.index = list(ic_labels.keys())
        return data

    def load_recording(self, montage, ica, ic_labels):
        """Load the object to be plotted."""
        data = self.init_vars(montage, ica, ic_labels)
        super(TopoVizICA, self).set_data(montage, data, None)
