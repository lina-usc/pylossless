from pathlib import Path

import dash
from dash import dcc, html
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


axis_template_time = dict(
    showspikes='across+toaxis',
    spikedash='solid',
    spikemode='across',
    spikesnap='cursor',
    showline=True,
    showgrid=True,
    zeroline=False,
    showbackground=False,
    backgroundcolor="rgb(200, 200, 230)",
    gridcolor="white",
    zerolinecolor="white")

axis_template = dict(
    showspikes=None,
    title="",
    zeroline=False,
    showline=False,
    showgrid=False,
    showticklabels=False,
    showlabel=False,
    showbackground=True,
    backgroundcolor="rgb(0, 0, 0)",
    gridcolor="rgb(0, 0, 0)",
    zerolinecolor="rgb(0, 0, 0)",
    tickmode='array',
    tickvals=[])



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
        self.ystep = 1e-3
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
        self.raw.load_data().filter(0.1, 40)

    def change_dir(self, directory):
        print(Path(directory))
        self.fname = list(Path(directory).glob('*.vhdr'))[0]  # TODO make permanent solution. cant assume vhdr all the time.
        self.load_raw()

    def initialize_layout(self):
        start, stop = self.raw.time_as_index([self.tmin, self.tmax])
        data, times = self.raw[:self.n_sel_ch, start:stop]
        self.layout = go.Layout(
                                width = 1500,
                                height=600,
                                xaxis={'zeroline': False,
                                       'showgrid': False},
                                yaxis={'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'autorange': False,  #'reversed',
                                       'scaleratio': 0.5,
                                       "tickmode": "array","tickvals": [],
                                       'range':[-self.ystep*self.n_sel_ch, self.ystep]},
                                showlegend=False)
        trace_kwargs = {'mode':'lines', 'line':dict(color='black', width=1)}

        # create objects for layout and traces
        self.traces = [] #[go.Scatter(x=times, y=data.T[:, 0], **trace_kwargs)]

        # loop over the channels
        for ii in range(self.n_sel_ch):
            self.traces.append(go.Scatter(x=times, y=data.T[:, ii] - ii * self.ystep, **trace_kwargs)) #yaxis=f'y{ii + 1}'
            #self.layout.update({f'y{ii + 1}':dict(layer='above traces', overlaying='y', showticklabels=False)})
        self.timeseries_graph.figure['layout'] = self.layout
        self.timeseries_graph.figure['data'] = self.traces

    def update_layout(self, ch_slider_val=None):
        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val

        first_sel_ch = self.ch_slider_val
        last_sel_ch = self.ch_slider_val + self.n_sel_ch

        start_samp, stop_samp = self.raw.time_as_index([self.tmin, self.tmax])

        ch_names = self.raw.ch_names[first_sel_ch:last_sel_ch]
                                     # basically names of first 20 channels

        data, times = self.raw[first_sel_ch:last_sel_ch, start_samp:stop_samp]

        # update the trace  ... couldu  do trace.x in trace?
        for i, (signal, trace) in enumerate(zip(data, self.traces)):
            trace.x = times
            trace.y = signal - i * self.ystep
        self.timeseries_graph.figure['data'] = self.traces
        # add channel names using Annotations
        '''annotations = go.Annotations([go.Annotation(x=-0.06, y=0, xref='paper', yref=f'y{ii + 1}',
                                                    text=ch_name, font=go.Font(size=9), showarrow=False)
                                      for ii, ch_name in enumerate(ch_names)])
        self.layout.update(annotations=annotations)

        # set the size of the figure and plot it
        self.layout.update(autosize=False, width=1000, height=600)'''


def NamedSlider(name, style=None, **kwargs):
    if style is None:
        style={"padding": "20px 10px 25px 4px"}
    return html.Div(
        style=style,
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
    eeg_visualizer.ch_slider_val = (len(eeg_visualizer.raw.ch_names) - eeg_visualizer.n_sel_ch) - value 
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
    NamedSlider(name="Channel",
                id="slider-channel",
                min=0,
                max=len(eeg_visualizer.raw.ch_names) - eeg_visualizer.n_sel_ch,
                step=1,
                marks=None,
                value=len(eeg_visualizer.raw.ch_names) - eeg_visualizer.n_sel_ch,
                included=False,
                updatemode='mouseup',
                vertical=True)
])


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
