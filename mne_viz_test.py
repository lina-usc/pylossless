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
raw.set_annotations(mne.Annotations(onset=[2.], duration=[1.], description=['test_annot'], orig_time=raw.info['meas_date']))


ica_fpath = Path("./tmp_test_files\derivatives\pylossless\sub-00\eeg\sub-00_task-test_ica2.fif")
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
    ica_visualizer.refresh_annotations()
    eeg_visualizer.refresh_annotations()


app = dash.Dash(__name__)
app.layout = html.Div([])

ica_dash_ids = {'graph':'graph-ica',
                'ch-slider':'ch-slider-ica',
                'time-slider':'time-slider-ica',
                'container-plot':'container-plot-ica',
                'keyboard':'keyboard-ica',
                'output':'output-ica'}
ica_visualizer = MNEVisualizer(app, raw_ica, dash_ids=ica_dash_ids, annot_created_callback=annot_created_callback)
eeg_visualizer = MNEVisualizer(app, raw, time_slider=ica_visualizer.dash_ids['time-slider'], 
                               dcc_graph_kwargs=dict(config={'modeBarButtonsToRemove':['zoom','pan']}),
                               annot_created_callback=annot_created_callback)

ica_visualizer.new_annot_desc = 'bad_manual'
eeg_visualizer.new_annot_desc = 'bad_manual'


#############################################################################
#############################################################################
#############################################################################


############################################################################
import warnings
from mne import create_info
from mne.io import RawArray
from mne.viz.topomap import _add_colorbar
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_values_topomap(value_dict, montage, axes, colorbar=True, cmap='RdBu_r',
                        vmin=None, vmax=None, names=None, image_interp='cubic', side_cb="right",
                        sensors=True, show_names=True, **kwargs):
    if names is None:
        names = montage.ch_names

    info = create_info(names, sfreq=256, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RawArray(np.zeros((len(names), 1)), info, copy=None, verbose=False).set_montage(montage)

    im = plot_topomap([value_dict[ch] for ch in names], pos=info, show=False, image_interp=image_interp,
                      sensors=sensors, res=64, axes=axes, names=names, show_names=show_names,
                      vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    if colorbar:
        try:
            cbar, cax = _add_colorbar(axes, im[0], cmap, pad=.05,
                                      format='%3.2f', side=side_cb)
            axes.cbar = cbar
            cbar.ax.tick_params(labelsize=12)

        except TypeError:
            pass

    return im
###########################################################################################################

from itertools import product
import plotly.express as px
from plotly.subplots import make_subplots
plt.switch_backend('agg')

rows = 6
cols = 4
ply_fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.01, 
                        vertical_spacing=0.01)

montage = raw.get_montage()

margin_x = 10
margin_y = 1
offset = 0

for no, (i, j) in enumerate(product(np.arange(rows), np.arange(cols))):
    component = ica.get_components()[:, no+offset]
    value_dict = dict(zip(ica.ch_names, component))

    fig, ax = plt.subplots(dpi=25)
    plot_values_topomap(value_dict, montage, ax, colorbar=False, show_names=False, names=ica.ch_names)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[margin_y:-margin_y, margin_x:-margin_x, :]
    px_fig = px.imshow(data)

    ply_fig.add_trace(px_fig.data[0], row=i+1, col=j+1)

for i in range (1, rows*cols+1):
  ply_fig['layout'][f'xaxis{i}'].update(showticklabels=False)
  ply_fig['layout'][f'yaxis{i}'].update(showticklabels=False)


ply_fig.update_layout(
    autosize=False,
    width=600,
    height=800,)
ply_fig['layout'].update(margin=dict(l=0,r=0,b=0,t=0))

#ply_fig.show()

#############################################################################
#############################################################################
#############################################################################




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
                                                   html.Div(children=[dcc.Graph(figure=ply_fig)],
                                                            )
                                                   ]
                                        )
                                    ],
                                style={'display':'block'})
                        ], style={"display":"block"})

app.layout.children.append(layout)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
