from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html  # deprecated
from dash.dependencies import Input, Output, State

# file selectioni
import tkinter
from tkinter import filedialog

# time series plot
from plotly import tools
import plotly.graph_objects as go
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font

# for card creation in time series plot
from dash import html

# loading raw object
from mne_bids import BIDSPath, read_raw_bids, get_bids_path_from_fname
import mne

class EEGVisualizer:

    def __init__(self, directory):
        self.fname = None
        self.raw = None
        self.bids_path = None
        self.raw = None
        self.change_dir(directory)  # sets self.bids_path, self.fname
        self.n_sel_ch = 20  # n of channels to display in plot
        self.tmin = 0  # min time to disp on plot
        self.tmax = 10  # max time to disp on plot
        self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        self.timeseries_graph = dcc.Graph(id='eeg_ts',
                                          figure={'data': None,
                                                  'layout': None})
        self.timeseries_div = html.Div([self.timeseries_graph],
                                       className="six columns")
        self.initialize_layout()

    @property
    def ch_slider_val(self):
        return self._ch_slider_val
    
    @ch_slider_val.setter
    def ch_slider_val(self, value):
        self.update_layout(value)

    def load_raw(self):
        self.bids_path = get_bids_path_from_fname(self.fname)
        self.raw = read_raw_bids(self.bids_path).pick('eeg')

    def change_dir(self, directory):
        print(Path(directory))
        self.fname = list(Path(directory).glob('*.vhdr'))[0]  # TODO make permanent solution. cant assume vhdr all the time.
        self.load_raw()

    def initialize_layout(self):
        start, stop = self.raw.time_as_index([self.tmin, self.tmax])
        data, times = self.raw[:self.n_sel_ch, start:stop]
        step = 1. / self.n_sel_ch
        kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

        # create objects for layout and traces
        self.layout = go.Layout(yaxis=go.YAxis(kwargs), showlegend=False)
        self.traces = [go.Scatter(x=times, y=data.T[:, 0])]

        # loop over the channels
        for ii in range(1, self.n_sel_ch):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            self.layout.update({f'yaxis{ii + 1}': go.YAxis(kwargs), 'showlegend': False})
            self.traces.append(go.Scatter(x=times, y=data.T[:, ii], yaxis=f'y{ii + 1}'))
        self.timeseries_graph.figure['layout'] = self.layout
        self.timeseries_graph.figure['data'] = go.Data(self.traces)

    def update_layout(self, ch_slider_val=None):
        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val

        ch_names = self.raw.ch_names[self.ch_slider_val:
                                     self.ch_slider_val +
                                     self.n_sel_ch]  # basically names of first 20 channels

        start, stop = self.raw.time_as_index([self.tmin, self.tmax])
        data, times = self.raw[:self.n_sel_ch, start:stop]

        # update the trace  ... couldu  do trace.x in trace?
        for signal, trace, in zip(data, self.traces):
            trace.x = times
            trace.y = signal
        self.timeseries_graph.figure['data'] = go.Data(self.traces)

        # add channel names using Annotations
        annotations = go.Annotations([go.Annotation(x=-0.06, y=0, xref='paper', yref=f'y{ii + 1}',
                                                    text=ch_name, font=go.Font(size=9), showarrow=False)
                                      for ii, ch_name in enumerate(ch_names)])
        self.layout.update(annotations=annotations)

        # set the size of the figure and plot it
        self.layout.update(autosize=False, width=1000, height=600)


def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )


app = dash.Dash(__name__)
eeg_visualizer = EEGVisualizer('./sub-01/ses-01/eeg')
server = app.server

@app.callback(
    Output('eeg_ts', 'figure'),
    Input('slider-channel', 'value'),
    #State('eeg_ts', 'figure')
)
def channel_slider_change(value):
    global eeg_visualizer
    eeg_visualizer.ch_slider_val = value
    return eeg_visualizer.timeseries_graph.figure # go.Data(eeg_visualizer.traces)


@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    # State('input-on-submit', 'value')
)
def _select_folder(n_clicks):
    global eeg_vizualizer
    if n_clicks:
        root = tkinter.Tk()
        root.withdraw()
        directory = filedialog.askdirectory()
        print('selected directory: ', directory)
        root.destroy()
        eeg_visualizer.change_dir(directory)
        return directory


app.layout = html.Div([
    html.Div([
        html.Button('Submit', id='submit-val'),
        html.Div(id='container-button-basic',
                 children='Enter a value and press submit')
            ]),
    html.Div([eeg_visualizer.timeseries_div,
              html.Div([]),
              html.Div([])
              ]),
    NamedSlider(name="Time",
                id="slider-channel",
                min=0,
                max=len(eeg_visualizer.raw.ch_names) - 20,
                step=1,
                marks={},
                value=0)
])


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
