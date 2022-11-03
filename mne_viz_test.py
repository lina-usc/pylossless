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

from mne_visualizer import MNEVisualizer

project_root = Path('./tmp_test_files')
derivatives_dir = project_root / 'derivatives'
files_list = [{'label':str(file), 'value':str(file)} for file in sorted(derivatives_dir.rglob("*.edf"))]


raw_graph_kwargs = dict(id='eeg_ts',
                        className='dcc-graph-timeseries',
                        figure={'data': None,
                                'layout': None})
comp_graph_kwargs = dict(id='comp_ts',
                         className='dcc-graph-timeseries',
                         figure={'data':None, 'layout': None})


####################
#  Begin Dash App
####################

from pathlib import Path

directory = './tmp_test_files/derivatives/pylossless/sub-00/eeg/'
fname = list(Path(directory).glob('*.edf'))[0] 
bids_path = get_bids_path_from_fname(fname)
raw = read_raw_bids(bids_path).pick('eeg')

app = dash.Dash(__name__)
#eeg_visualizer = EEGVisualizer('./tmp_test_files/derivatives/pylossless/sub-00/eeg') #./sub-01/ses-01/eeg

ica_dash_ids = {'graph':'graph-ica',
                'ch-slider':'ch-slider-ica',
                'time-slider':'time-slider-ica',
                'container-plot':'container-plot-ica'}
ica_visualizer = MNEVisualizer(app, raw, dash_ids=ica_dash_ids)
eeg_visualizer = MNEVisualizer(app, raw, time_slider=ica_visualizer.dash_ids['time-slider'])



server = app.server



####################
# Callbacks
####################

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


########################
# Layout
########################


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
                                         children=[html.Div([eeg_visualizer.container_plot,
                                                   ica_visualizer.container_plot]),
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
