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

project_root = Path('./tmp_test_files')
derivatives_dir = project_root / 'derivatives'
files_list = [{'label':str(file), 'value':str(file)} for file in sorted(derivatives_dir.rglob("*.edf"))]

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

raw_graph_kwargs = dict(id='eeg_ts',
                        className='dcc-graph-timeseries',
                        figure={'data': None,
                                'layout': None})
comp_graph_kwargs = dict(id='comp_ts',
                         className='dcc-graph-timeseries',
                         figure={'data':None, 'layout': None})

class EEGVisualizer:

    def __init__(self, directory):
        self.fname = None
        self.raw = None
        self.ica = None
        self.ica_sources = None
        self.bids_path = None
        self.raw = None
        self.change_dir(directory)  # sets self.bids_path, self.fname
        self.load_ica() # loads ica
        self.n_sel_ch = 20  # n of channels to display in plot
        self.win_start = 0  # min time to disp on plot
        self.win_size = 10  # max time to disp on plot
        self.ystep = 1e-3
        self.comp_ystep = 1e1
        self.layout = None
        self.traces = None
        self._ch_slider_val = 0
        #self._time_slider_val = 0
        self.timeseries_graph = dcc.Graph(**raw_graph_kwargs)
        self.timeseries_div = html.Div([self.timeseries_graph],
                                        className='timeseries-div')  # className="six columns",
        self.comp_ts_graph = dcc.Graph(**comp_graph_kwargs)
        self.comp_ts_div = html.Div([self.comp_ts_graph], className='timeseries-div')  # className="six columns"
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
        self.raw.load_data() #.filter(0.1, 40)

    def load_ica(self):
        ica_fname = self.bids_path.fpath.stem[:-4] + "_ica2"
        ica_path = self.bids_path.fpath.with_name(ica_fname).with_suffix('.fif')
        self.ica = mne.preprocessing.read_ica(ica_path)
        self.ica_sources = self.ica.get_sources(self.raw)

    def change_dir(self, directory):
        print(Path(directory))
        self.fname = list(Path(directory).glob('*.edf'))[0]  # TODO make permanent solution. cant assume vhdr all the time.
        self.load_raw()

############################
# Create Timeseries Layouts
############################

    def initialize_layout(self):
        start, stop = self.raw.time_as_index([self.win_start, self.win_size])
        data, times = self.raw[:self.n_sel_ch, start:stop]
        comp_data, _ = self.ica_sources[:self.n_sel_ch, start:stop]

        self.layout = go.Layout(
                                width = 1200,
                                height=400,
                                xaxis={'zeroline': False,
                                       'showgrid': True,
                                       'title': "time (seconds)",
                                       'gridcolor':'white'},
                                yaxis={'showgrid': True,
                                       'showline': True,
                                       'zeroline': False,
                                       'autorange': False,  #'reversed',
                                       'scaleratio': 0.5,
                                       "tickmode": "array",
                                       "tickvals": np.arange(-self.n_sel_ch + 1, 1) * self.ystep,
                                       'ticktext': [''] * self.n_sel_ch,
                                       'range':[-self.ystep*self.n_sel_ch, self.ystep]},
                                showlegend=False,
                                margin={'t': 25,'b': 25,'l': 35, 'r': 25},
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#EAEAF2",
                                )

        trace_kwargs = {'mode':'lines', 'line':dict(color='#222222', width=1)}
        # create objects for layout and traces
        self.traces = [] #[go.Scatter(x=times, y=data.T[:, 0], **trace_kwargs)]
        self.comp_traces = []

        # loop over the channels
        for ii in range(self.n_sel_ch):
            self.traces.append(go.Scatter(x=times, y=data.T[:, ii] - ii * self.ystep, **trace_kwargs)) #yaxis=f'y{ii + 1}'
            #self.layout.update({f'y{ii + 1}':dict(layer='above traces', overlaying='y', showticklabels=False)})
            self.comp_traces.append(go.Scatter(x=times,
                                               y=comp_data.T[:, ii] - ii * self.comp_ystep,
                                               **trace_kwargs)
                                    )

        self.timeseries_graph.figure['layout'] = self.layout
        self.timeseries_graph.figure['data'] = self.traces

        self.comp_ts_graph.figure['layout'] = go.Layout(self.layout)
        self.comp_ts_graph.figure['layout'].yaxis['range'] = [-self.comp_ystep*self.n_sel_ch, self.comp_ystep]
        self.comp_ts_graph.figure['layout'].yaxis['tickvals'] = np.arange(-self.n_sel_ch + 1, 1) * self.comp_ystep
        self.comp_ts_graph.figure['data'] = self.comp_traces

    def update_layout(self,
                      ch_slider_val=None,
                      component_slider_val=None,
                      time_slider_val=None):
    
        if ch_slider_val is not None:
            self._ch_slider_val = ch_slider_val
        if component_slider_val is not None:
            self._component_slider_val = component_slider_val
        if time_slider_val is not None:
            self.win_start = time_slider_val

        # Update selected channels
        first_sel_ch = self._ch_slider_val #self.ch_slider_val
        last_sel_ch = self._ch_slider_val + self.n_sel_ch #self.ch_slider_val + self.n_sel_ch

        # Update selected ICA components
        # first_sel_comp = self._component_slider_val
        # last_sel_comp = self._component_slider_val + self.n_sel_ch

        # Update times
        start_samp, stop_samp = self.raw.time_as_index([self.win_start, self.win_start + self.win_size])

        data, times = self.raw[first_sel_ch:last_sel_ch, start_samp:stop_samp]

        # comp_data, _ = self.ica_sources[:self.n_sel_ch, start_samp:stop_samp]


        # Update the raw timeseries traces
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

        # Update the component timeseries traces
        '''comp_names = self.ica_sources.ch_names[first_sel_comp:last_sel_comp]
        self.comp_ts_graph.layout.yaxis['ticktext'] = comp_names[::-1]
        for i, (comp_name, signal, trace) in enumerate(zip(comp_names, comp_data, self.comp_traces)):
            trace.x = times
            trace.y = signal - i * self.comp_ystep
            trace.name = comp_name
        self.comp_ts_graph.figure['data'] = self.comp_traces'''


####################
#  Begin Dash App
####################

app = dash.Dash(__name__)
#eeg_visualizer = EEGVisualizer('./tmp_test_files/derivatives/pylossless/sub-00/eeg') #./sub-01/ses-01/eeg
eeg_visualizer = EEGVisualizer(app, raw) #./sub-01/ses-01/eeg

server = app.server



####################
# Callbacks
####################

@app.callback(
    Output('eeg_ts', 'figure'),
    Input('slider-channel', 'value'),
    Input('slider-time', 'value')
    #State('eeg_ts', 'figure')
)
def channel_slider_change(value_ch, value_time):
    global eeg_visualizer
    value_ch -= len(eeg_visualizer.raw.ch_names) -1
    #eeg_visualizer.ch_slider_val = (len(eeg_visualizer.raw.ch_names) - eeg_visualizer.n_sel_ch) - value_ch
    #eeg_visualizer.ch_slider_time = value_time  # (eeg_visualizer.raw.times[-1] - eeg_visualizer.win_size) - 
    eeg_visualizer.update_layout(ch_slider_val=value_ch, time_slider_val=value_time)
    return eeg_visualizer.timeseries_graph.figure # go.Data(eeg_visualizer.traces)

'''@app.callback(
              Output('comp_ts', 'figure'),
              Input('slider-ica_source', 'value'),
              Input('slider-time', 'value')
              )
def ica_source_slider_change(value_ica_source, value_time):
    global eeg_visualizer
    value_ica_source -= len(eeg_visualizer.ica_sources.ch_names) -1
    eeg_visualizer.update_layout(component_slider_val=value_ica_source, time_slider_val=value_time)
    return eeg_visualizer.comp_ts_graph.figure  # go.Data(eeg_visualizer.traces)'''

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

####################
# Create dcc.Slider components
####################

def NamedSlider(name, style=None, **slider_kwargs):
    if style is None:
        style={'display': 'inline-block'}
    return html.Div(
                    style=style,
                    children=[
                            html.Div(children=dcc.Slider(**slider_kwargs),
                                    style={"margin-left": "6px"}),
                            html.P(f"{name}:", style={"color":"#CBD2D9",
                                                      "textAlign": "left"})
                            ],
                    )

channel_slider = NamedSlider(name="",
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
                          style= {'width': '1200px'},
                          id="slider-time",
                          min=eeg_visualizer.raw.times[0],
                          max=eeg_visualizer.raw.times[-1] - eeg_visualizer.win_size,
                          marks={int(key):str(int(key)) for key in marks_keys} ,#dict(zip(marks_keys,marks_keys.astype(str))),
                          value=eeg_visualizer.win_start,
                          vertical=False,
                          included=False,
                          updatemode='drag',  # updates while moving slider
                          )


source_slider = NamedSlider(name="",
                            id="slider-ica_source",
                            min=eeg_visualizer.n_sel_ch,
                            max=len(eeg_visualizer.ica_sources.ch_names) -1,
                            step=1,
                            marks=None,
                            value=len(eeg_visualizer.ica_sources.ch_names) -1,
                            included=False,
                            updatemode='mouseup',
                            vertical=True,
                            verticalHeight=300
                            )

########################
# Layout
########################

file_browser_css = {'display':'inline-block',
                    'margin':'0px 5px 5px 5px',
                    'padding':'0px 0px 0px 2px'
                    }
dropdown_css = {'width':700,
                'height':30,
                'display':'inline-block',
                'margin':'2px 0px 0px 2px',
                'padding':'5px 2px 0px 2px'}

app.layout = html.Div([
                        html.Div([
                                    html.Button('Folder',
                                                id='submit-val',
                                                className="folderButton",
                                                title=f'current folder: {project_root.resolve()}'
                                                ),
                                    dcc.Dropdown(id="fileDropdown",
                                                className="card",
                                                options=files_list,
                                                placeholder="Select a file"
                                                ),
                                    html.Div(id='container-button-basic',
                                            children='Enter a value and press submit')
                                    ],
                                    className='banner'
                                 ),
                        html.Div([
                                html.Div(id='plots-container', 
                                         children=[
                                                   html.Div([
                                                             html.Div([channel_slider, eeg_visualizer.timeseries_div], className="outer-timeseries-div"),
                                                             html.Div([source_slider, eeg_visualizer.comp_ts_div], className="outer-timeseries-div", id='component_div'),
                                                             time_slider],
                                                             className='timeseries-container',
                                                            ),
                                                   html.Div([], style={
                                                                    'border':'2px solid red',
                                                                    'width':'25%',
                                                                    'height':'50%'
                                                                    }
                                                            )
                                                   ]
                                        )
                                    ],
                                style={'display':'block'})
                        ], style={"display":"block"})


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
