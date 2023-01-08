from itertools import product
import warnings

import plotly.express as px
from plotly.subplots import make_subplots
from dash import dcc, html, no_update
from dash.dependencies import Input, Output

import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne import create_info
from mne.io import RawArray
from mne.viz.topomap import _add_colorbar
from mne.viz import plot_topomap


def plot_values_topomap(value_dict, montage, axes, colorbar=True, cmap='RdBu_r',
                        vmin=None, vmax=None, names=None, image_interp='cubic',
                        side_cb="right", sensors=True,
                        **kwargs):
    if names is None:
        names = montage.ch_names

    info = create_info(names, sfreq=256, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RawArray(np.zeros((len(names), 1)), info, copy=None,
                 verbose=False).set_montage(montage)

    im = plot_topomap([value_dict[ch] for ch in names], pos=info, show=False,
                       image_interp=image_interp, sensors=sensors, res=64,
                       axes=axes, names=names,vlim=(vmin, vmax), cmap=cmap, **kwargs)

    if colorbar:
        try:
            cbar = _add_colorbar(axes, im[0], cmap, pad=.05,
                                 format='%3.2f', side=side_cb)[0]
            axes.cbar = cbar
            cbar.ax.tick_params(labelsize=12)

        except TypeError:
            pass

    return im


class TopoData:
    def __init__(self, topo_values=()):
        """topo_values: list of dict """
        self.topo_values = list(topo_values)

    def add_topomap(self, topomap: dict):
        """topomap: dict"""
        self.topo_values.append(topomap)

    @property
    def nb_topo(self):
        return len(self.topo_values)


class TopoViz:
    def __init__(self, app, montage, data: TopoData, rows=5, cols=4, margin_x=10,
                 width=600, height=800, margin_y=1, topo_slider_id=None):
        """ """
        self.montage = montage
        self.data = data
        self.app = app
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        fig = make_subplots(rows=rows, cols=cols,
                                     horizontal_spacing=0.01, 
                                     vertical_spacing=0.01)
        self.graph = dcc.Graph(figure=fig, id='topo-graph', className='dcc-graph') # 
        self.graph_div = html.Div(children=[self.graph],
                                  style={"border":"2px red solid"},
                                  className='dcc-graph-div')

        self.margin_x = margin_x
        self.margin_y = margin_y
        self.offset = 0
        self.topo_slider = None
        self.use_topo_slider = topo_slider_id
        

        self.init_slider()
        
        self.initialize_layout()
        self.set_div()
        self.set_callback()

    def get_topo_data(self, i, j):
        no = i*self.cols + j
        if no + self.offset >= self.data.nb_topo: # out of range
            return [[]]
        value_dict = self.data.topo_values[no+self.offset]

        fig, ax = plt.subplots(dpi=25)
        plot_values_topomap(value_dict, self.montage, ax, colorbar=False,
                            names=list(value_dict.keys()))  
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data[self.margin_y:-self.margin_y,
                    self.margin_x:-self.margin_x, :]

    def initialize_layout(self):
        for i, j in product(np.arange(self.rows), np.arange(self.cols)):
            self.graph.figure.add_trace(px.imshow(self.get_topo_data(i, j)).data[0], 
                                        row=i+1, col=j+1)

        for i in range(1, self.rows*self.cols+1):
            self.graph.figure['layout'][f'xaxis{i}'].update(showticklabels=False)
            self.graph.figure['layout'][f'yaxis{i}'].update(showticklabels=False)

        self.graph.figure.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,)
        self.graph.figure['layout'].update(margin=dict(l=0,r=0,b=0,t=0))

    def update_layout(self, slider_val):
        self.offset = self.topo_slider.max-slider_val
        for no, (i, j) in enumerate(product(np.arange(self.rows),
                                            np.arange(self.cols))):

            if no + self.offset >= self.data.nb_topo: # out of range
                break

            self.graph.figure.update_traces(z=self.get_topo_data(i, j), row=i+1, col=j+1)

    def init_slider(self):
        self.topo_slider = dcc.Slider(id='topo-slider',
                                min=self.rows * self.cols -1,
                                max=self.data.nb_topo -1,
                                step=1,
                                marks=None,
                                value=self.data.nb_topo -1,
                                included=False,
                                updatemode='mouseup',
                                vertical=True,
                                verticalHeight=self.graph.figure.layout.height) #300)

        
    def set_div(self):
        if self.use_topo_slider is None:
            # outer_div includes slider obj
            outer_div = [html.Div(self.topo_slider, style={"border":"2px purple solid", 'display':'inline-block'}), self.graph_div]
        else:
            # outer_div is just the graph
            outer_div = [self.graph_div]
        self.container_plot = html.Div(children=outer_div, style={"border":"2px black solid"}, className="outer-timeseries-div")

    def set_callback(self):
        args = [Output('topo-graph', 'figure')]
        if self.use_topo_slider:
            args += [Input(self.use_topo_slider, 'value')]
        else: 
            args += [Input('topo-slider', 'value')]

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(slider_val):      
            print("update layout", slider_val)            
            self.update_layout(slider_val=slider_val)
            return self.graph.figure


class TopoVizICA(TopoViz):
    def __init__(self, app, inst, ica, **kwargs):
        """ """
        self.data = TopoData([dict(zip(ica.ch_names, component))
                              for component in ica.get_components().T])
        super(TopoVizICA, self).__init__(app, inst, self.data, **kwargs)
