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

import numpy as np

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
    showticklabels=True,
    showlabel=True,
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
        self.win_start = 0  # min time to disp on plot
        self.win_size = 10  # max time to disp on plot
        self.ystep = 1e-3
        self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        #self._time_slider_val = 0
        self.timeseries_graph = dcc.Graph(id='eeg_ts',
                                          figure={'data': None,
                                                  'layout': None},
                                          style={"border":"2px dotted blue"})
        self.timeseries_div = html.Div([self.timeseries_graph],
                                       className="six columns", style={"border":"5px solid red", 'width':'50%'})
        self.initialize_layout()

    '''@property
    def ch_slider_val(self):
        return self._ch_slider_val

    @ch_slider_val.setter
    def ch_slider_val(self, value):
        self.update_layout(ch_slider_val=value)

    @property
    def time_slider_val(self):
        return self._time_slider_val

    @time_slider_val.setter
    def time_slider_val(self, value):
        self.update_layout(time_slider_val=value)'''

    def load_raw(self):
        self.bids_path = get_bids_path_from_fname(self.fname)
        self.raw = read_raw_bids(self.bids_path).pick('eeg')
        self.raw.load_data().filter(0.1, 40)

    def change_dir(self, directory):
        print(Path(directory))
        self.fname = list(Path(directory).glob('*.vhdr'))[0]  # TODO make permanent solution. cant assume vhdr all the time.
        self.load_raw()

    def initialize_layout(self):
        start, stop = self.raw.time_as_index([self.win_start, self.win_size])
        data, times = self.raw[:self.n_sel_ch, start:stop]
        self.layout = go.Layout(
                                width = 1200,
                                height=400,
                                xaxis={'zeroline': False,
                                       'showgrid': False,
                                       'title': "time (seconds)"},
                                yaxis={'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'autorange': False,  #'reversed',
                                       'scaleratio': 0.5,
                                       "tickmode": "array",
                                       "tickvals": np.arange(-self.n_sel_ch + 1, 1) * self.ystep,
                                       'ticktext': [''] * self.n_sel_ch,
                                       'range':[-self.ystep*self.n_sel_ch, self.ystep]},
                                showlegend=False,
                                margin={'t':5,'b':25,'r':5},
                                paper_bgcolor="LightSteelBlue",
                                plot_bgcolor="LightSteelBlue"
                                )
        trace_kwargs = {'mode':'lines', 'line':dict(color='black', width=1)}

        # create objects for layout and traces
        self.traces = [] #[go.Scatter(x=times, y=data.T[:, 0], **trace_kwargs)]

        # loop over the channels
        for ii in range(self.n_sel_ch):
            self.traces.append(go.Scatter(x=times, y=data.T[:, ii] - ii * self.ystep, **trace_kwargs)) #yaxis=f'y{ii + 1}'
            #self.layout.update({f'y{ii + 1}':dict(layer='above traces', overlaying='y', showticklabels=False)})
        self.timeseries_graph.figure['layout'] = self.layout
        self.timeseries_graph.figure['data'] = self.traces

    def update_layout(self, ch_slider_val=None, time_slider_val=None):
        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val
        if time_slider_val is not None:
            self.win_start = time_slider_val
        first_sel_ch = self._ch_slider_val #self.ch_slider_val
        last_sel_ch = self._ch_slider_val + self.n_sel_ch #self.ch_slider_val + self.n_sel_ch

        start_samp, stop_samp = self.raw.time_as_index([self.win_start, self.win_start + self.win_size])

        data, times = self.raw[first_sel_ch:last_sel_ch, start_samp:stop_samp]

        # update the trace  ... couldu  do trace.x in trace?
        ch_names = self.raw.ch_names[first_sel_ch:last_sel_ch]
        self.layout.yaxis['ticktext'] = ch_names[::-1]
        for i, (ch_name, signal, trace) in enumerate(zip(ch_names, data, self.traces)):
            trace.x = times
            trace.y = signal - i * self.ystep
            trace.name = ch_name
        self.timeseries_graph.figure['data'] = self.traces
        # add channel names using Annotations
        '''annotations = go.Annotations([go.Annotation(x=-0.06, y=0, xref='paper', yref=f'y{ii + 1}',
                                                    text=ch_name, font=go.Font(size=9), showarrow=False)
                                      for ii, ch_name in enumerate(ch_names)])
        self.layout.update(annotations=annotations)

        # set the size of the figure and plot it
        self.layout.update(autosize=False, width=1000, height=600)'''


def NamedSlider(name, style=None, **slider_kwargs):
    if style is None:
        style={"padding": "0px 10px 25px 4px"}
    return html.Div(
        style=style,
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**slider_kwargs)),
        ],
    )


app = dash.Dash(__name__)
eeg_visualizer = EEGVisualizer('./sub-01/ses-01/eeg')
server = app.server

@app.callback(
    Output('eeg_ts', 'figure'),
    Input('slider-channel', 'value'),
    Input('slider-time', 'value')
    #State('eeg_ts', 'figure')
)
def channel_slider_change(value_ch, value_time):
    global eeg_visualizer
    print(f'value_ch: {value_ch}, value time: {value_time}')
    value_ch -= len(eeg_visualizer.raw.ch_names) -1
    #eeg_visualizer.ch_slider_val = (len(eeg_visualizer.raw.ch_names) - eeg_visualizer.n_sel_ch) - value_ch
    #eeg_visualizer.ch_slider_time = value_time  # (eeg_visualizer.raw.times[-1] - eeg_visualizer.win_size) - 
    eeg_visualizer.update_layout(ch_slider_val=value_ch, time_slider_val=value_time)
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



channel_slider = NamedSlider(name="Channel",
                            id="slider-channel",
                            min=eeg_visualizer.n_sel_ch,
                            max=len(eeg_visualizer.raw.ch_names) -1,
                            step=1,
                            marks=None,
                            value=len(eeg_visualizer.raw.ch_names) -1,
                            included=False,
                            updatemode='mouseup',
                            vertical=True,
                            verticalHeight=300)

marks_keys = np.round(np.linspace(eeg_visualizer.raw.times[0], eeg_visualizer.raw.times[-1], 10))
time_slider = NamedSlider(name="Time",
                            id="slider-time",
                            min=eeg_visualizer.raw.times[0],
                            max=eeg_visualizer.raw.times[-1] - eeg_visualizer.win_size,
                            marks={int(key):str(int(key)) for key in marks_keys} ,#dict(zip(marks_keys,marks_keys.astype(str))),
                            value=eeg_visualizer.win_start,
                            vertical=False,
                            included=False,
                            updatemode='drag',  # updates while moving slider
                            style={"width":"1500px",
                                    "padding":"25px 0px 0px 0px",
                                    "marginTop":25}
                            )




file_browser_css = {'display':'inline-block',
                    'border':'2px dotted red',
                    'margin':'0px 5px 0px 5px',
                    'padding':'0px 2px 0px 2px'
                    }
dropdown_css = {'width':500,
                'height':30,
                'display':'inline-block',
                'border':'2px dotted black',
                'margin':'2px 0px 0px 2px',
                'padding':'5px 2px 0px 2px'}
project_dir = Path()

app.layout = html.Div([
                        html.Div([
                                  html.Button('Select folder',
                                              id='submit-val',
                                              title=f'current folder: {project_dir.resolve()}',
                                              style=file_browser_css),
                                  dcc.Dropdown(id="file_dropdown",
                                              options=file_dicts,
                                              placeholder="Select a file",
                                              style=dropdown_css),
                                  html.Div(id='container-button-basic',
                                children='Enter a value and press submit',
                                style={'border':'2px solid blue'})
                                ]
                               ),
                        html.Div([eeg_visualizer.timeseries_div, channel_slider], style={"border":"5px solid green"}),
                        time_slider,
                      ],style={"border":"5px solid black"})


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
