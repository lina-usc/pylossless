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
import matplotlib.pyplot as plt

# loading raw object
from mne_bids import BIDSPath, read_raw_bids, get_bids_path_from_fname
import mne

from mne_visualizer import MNEVisualizer
from topo_viz import TopoVizICA

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

'''directory = './tmp_test_files/derivatives/pylossless/sub-00/eeg/'
fname = list(Path(directory).glob('*.edf'))[0] 
bids_path = get_bids_path_from_fname(fname)
raw = read_raw_bids(bids_path).pick('eeg')
raw.set_annotations(mne.Annotations(onset=[2.5, 4],
                                    duration=[1., 0],
                                    description=['test_annot']*2,
                                    orig_time=raw.info['meas_date']))'''


fname = Path('../Face13/derivatives/pylossless/sub-s02/eeg/sub-s02_task-faceO_eeg.edf')
bids_path = get_bids_path_from_fname(fname)
raw = read_raw_bids(bids_path).pick('eeg')
raw.info['bads'].extend(['A27', 'A5', 'B10', 'B16', 'B17', 'B27', 'B28', 'C10',
                         'C17','C18', 'C3', 'D1',])


# ica_fpath = Path("./tmp_test_files\derivatives\pylossless\sub-00\eeg\sub-00_task-test_ica2.fif")
ica_fpath = Path('../Face13/derivatives/pylossless/sub-s02/eeg/sub-s02_task-faceO_ica2.fif')
ica = mne.preprocessing.read_ica(ica_fpath)
info = mne.create_info(ica._ica_names,
                       sfreq=raw.info['sfreq'],
                       ch_types=['eeg'] * ica.n_components_)

raw_ica = mne.io.RawArray(ica.get_sources(raw).get_data(), info)
raw_ica.set_meas_date(raw.info['meas_date'])
raw_ica.set_annotations(raw.annotations)



def annot_created_callback(annotation):
    raw.set_annotations(raw.annotations + annotation)
    raw_ica.set_annotations(raw_ica.annotations + annotation)
    ica_visualizer.update_layout(ch_slider_val=ica_visualizer.channel_slider.max,
                                 time_slider_val=ica_visualizer.win_start)
    eeg_visualizer.update_layout()


app = dash.Dash(__name__)
app.layout = html.Div([])

#raw.info['bads'].append('E5')
ica_visualizer = MNEVisualizer(app, raw_ica, dash_id_suffix='ica', annot_created_callback=annot_created_callback)
eeg_visualizer = MNEVisualizer(app, raw, time_slider=ica_visualizer.dash_ids['time-slider'], 
                               dcc_graph_kwargs=dict(config={'modeBarButtonsToRemove':['zoom','pan']}),
                               annot_created_callback=annot_created_callback)
ica_topo = TopoVizICA(app, raw.get_montage(), ica, topo_slider_id=ica_visualizer.dash_ids['ch-slider'])

ica_visualizer.new_annot_desc = 'bad_manual'
eeg_visualizer.new_annot_desc = 'bad_manual'

ica_visualizer.update_layout()


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


layout =     html.Div([
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
                                                   ica_topo.container_plot,
                                                   ],
                                        style={"border":"2px green solid"})
                                    ],
                                style={'display':'block'})
                        ], style={"display":"block"})

app.layout.children.append(layout)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
