from itertools import product
import warnings

from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
from copy import copy

from mne import create_info
from mne.io import RawArray

import plotly.graph_objects as go
from mne.viz.topomap import _get_pos_outlines
from mne.utils.check import _check_sphere
from mne.viz.topomap import _setup_interp, _check_extrapolate

from . import ic_label_cmap
from .css_defaults import CSS, STYLE

from copy import copy

axis = {'showgrid': False, # thin lines in the background
         'visible': False,  # numbers below
        }
yaxis = copy(axis)
yaxis.update(dict(scaleanchor="x", scaleratio=1))

class TopoData:
    def __init__(self, topo_values=()):
        """topo_values: list of dict """
        self.topo_values = pd.DataFrame(topo_values)

    def add_topomap(self, topomap: dict):
        """topomap: dict"""
        self.topo_values = self.topo_values.append(topomap)

    @property
    def nb_topo(self):
        return self.topo_values.shape[0]


class TopoViz:
    def __init__(self, app, montage=None, data=None,  # : TopoData,
                 rows=5, cols=4, margin_x=4/5, width=400, height=600,
                 margin_y=2/5, head_contours_color="black", cmap='RdBu_r',
                 show_sensors=True, show_slider=True, refresh_input=None):
        """ """
        self.refresh_input = refresh_input
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

    def load_recording(self, montage, data):
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

    def set_head_pos_contours(self, sphere=None, picks=None):
        if not self.info:
            return
        sphere = _check_sphere(sphere, self.info)
        self.pos, self.outlines = _get_pos_outlines(self.info, picks, sphere,
                                                    to_sphere=True)

    def get_head_scatters(self, color="back", show_sensors=True):
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
        else:
            return outlines_scat

    def get_heatmap_data(self, i, j, ch_type="eeg", res=64,
                         extrapolate='auto'):
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
        x_rad , y_rad = self.outlines["clip_radius"]
        Zi[np.sqrt(((Xi - x0)/x_rad)**2 + ((Yi-y0)/y_rad)**2) > 1] = np.nan

        return {"x": Xi[0], "y": Yi[:, 0], "z": Zi}

    def initialize_layout(self, slider_val=None, subplot_titles=None,
                          show_sensors=True):
        """ """
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
                                                show_sensors=self.show_sensors):
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
            plot_bgcolor='rgba(0,0,0,0)',  #'#EAEAF2', #'rgba(44,44,44,.5)',
            paper_bgcolor='rgba(0,0,0,0)')  #'#EAEAF2')  #'rgba(44,44,44,.5)')
        self.graph.figure['layout'].update(margin=dict(l=0, r=0, b=0, t=20))

    @property
    def nb_topo(self):
        if self.data:
            return self.data.nb_topo
        return self.rows * self.cols

    def init_slider(self):
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
        # outer_div includes slider obj
        graph_components = [self.topo_slider_div, self.graph_div]
        self.container_plot = html.Div(children=graph_components,
                                       className=CSS['topo-container'])

    def set_callback(self):
        args = [Output('topo-graph', 'figure')]
        args += [Input('topo-slider', 'value')]
        if self.refresh_input:
            args += [self.refresh_input]

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(slider_val, *args):
            self.initialize_layout(slider_val=slider_val,
                                   subplot_titles=self.titles,
                                   show_sensors=self.show_sensors)
            return self.graph.figure


class TopoVizICA(TopoViz):
    def __init__(self, app, montage, ica, ic_labels=None, **kwargs):
        """ """
        data = self.init_vars(montage, ica, ic_labels)
        super(TopoVizICA, self).__init__(app, montage, data, **kwargs)

    def init_vars(self, montage, ica, ic_labels):
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
        data = self.init_vars(montage, ica, ic_labels)
        super(TopoVizICA, self).load_recording(montage, data)
