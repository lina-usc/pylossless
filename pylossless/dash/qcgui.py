from pathlib import Path

# file selection
import tkinter
from tkinter import filedialog
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# loading raw object
from mne_bids import read_raw_bids, get_bids_path_from_fname
import mne

from .topo_viz import TopoVizICA
from .mne_visualizer import MNEVisualizer, ICVisualizer

from . import ic_label_cmap

from .css_defaults import CSS, STYLE


def get_ll_assets(fname, ica_fpath, iclabel_fpath='', bads=(),
                  verbose=False):
    # Validity check
    if not fname or not ica_fpath or not iclabel_fpath:
        return None, None, None, None

    if iclabel_fpath is not None:
        if not iclabel_fpath.exists():
            msg = (f'Could not find {iclabel_fpath.name}. Searched in'
                   f' {iclabel_fpath.parent.absolute()}')
            raise FileExistsError(msg)

    bids_path = get_bids_path_from_fname(fname, verbose=verbose)
    raw = read_raw_bids(bids_path, verbose=False).pick('eeg')
    raw.info['bads'].extend(bads)

    ica = mne.preprocessing.read_ica(ica_fpath, verbose=verbose)
    info = mne.create_info(ica._ica_names,
                        sfreq=raw.info['sfreq'],
                        ch_types=['eeg'] * ica.n_components_, verbose=verbose)

    raw_ica = mne.io.RawArray(ica.get_sources(raw).get_data(), info, verbose=verbose)
    raw_ica.set_meas_date(raw.info['meas_date'])
    raw_ica.set_annotations(raw.annotations)

    return raw, raw_ica, ica, iclabel_fpath


class QCGUI:

    def __init__(self, app, raw=None, raw_ica=None, ica=None,
                 iclabel_fpath=None, project_root=None):

        if project_root is None:
            project_root = Path(__file__).parent.parent.parent / 'assets' / 'test_data'
        self.project_root = Path(project_root)

        self.app = app
        self.raw = raw
        self.raw_ica = raw_ica
        self.ica = ica

        self.ica_visualizer = None
        self.eeg_visualizer = None
        self.ica_topo = None
        self.ic_types = None
        self.set_layout()
        self.set_callbacks()
        self.load_recording('c:/users/shuber10/documents/github_repos/pylossless/assets/test_data/sub-s02/eeg/sub-s02_task-faceO_eeg.edf')

    def annot_created_callback(self, annotation):
        self.raw.set_annotations(self.raw.annotations + annotation)
        self.raw_ica.set_annotations(self.raw_ica.annotations + annotation)
        self.ica_visualizer.update_layout(ch_slider_val=self.ica_visualizer
                                                            .channel_slider
                                                            .max,
                                          time_slider_val=self.ica_visualizer
                                                              .win_start)
        self.eeg_visualizer.update_layout()

    def set_visualizers(self):
        # Setting time-series and topomap visualizers
        if self.ic_types:
            cmap = {ic: ic_label_cmap[ic_type]
                    for ic, ic_type in self.ic_types.items()}
        else:
            cmap = None
        self.ica_visualizer = ICVisualizer(self.app, self.raw_ica,
                                           dash_id_suffix='ica',
                                           annot_created_callback=self.annot_created_callback,
                                           cmap=cmap,
                                           ic_types=self.ic_types)
        self.eeg_visualizer = MNEVisualizer(self.app,
                                            self.raw,
                                            time_slider=self.ica_visualizer
                                                            .dash_ids['time-slider'],
                                            dcc_graph_kwargs=dict(config={'modeBarButtonsToRemove':['zoom','pan']}),
                                            annot_created_callback=self.annot_created_callback)

        montage = self.raw.get_montage() if self.raw else None
        self.ica_topo = TopoVizICA(self.app, montage, self.ica, self.ic_types,
                                   topo_slider_id=self.ica_visualizer.dash_ids['ch-slider'],
                                   show_sensors=True)

        self.ica_visualizer.new_annot_desc = 'bad_manual'
        self.eeg_visualizer.new_annot_desc = 'bad_manual'

        self.ica_visualizer.update_layout()

    def set_layout(self):
        # app.layout must not be None for some of the operations of the
        # visualizers.
        self.app.layout = html.Div([])
        self.set_visualizers()

        # Layout for file control row
        # derivatives_dir = self.project_root / 'derivatives'
        files_list = []
        # files_list = [dbc.DropdownMenuItem(str(file.name))
        #              for file in sorted(self.project_root.rglob("*.edf"))]
        dropdown_text = f'current folder: {self.project_root.resolve()}'
        logo_fpath = '../assets/logo.png'
        folder_button = dbc.Button('Folder', id='folder-selector',
                                   color='primary',
                                   outline=True,
                                   className=CSS['button'],
                                   title=dropdown_text)
        save_button = dbc.Button('Save', id='save-button',
                                 color='info',
                                 outline=True,
                                 className=CSS['button'])
        '''drop_down = dbc.DropdownMenu(label="Select a file",
                                     id='file-dropdown',
                                     className=CSS['dropdown'],
                                     children=files_list
                                     )'''
        drop_down = dcc.Dropdown(id='file-dropdown',
                                 className=CSS['dropdown'],
                                 placeholder="Select a file",
                                 options=files_list)
        control_header_row = dbc.Row([
                                    dbc.Col([folder_button, save_button],
                                            width={'size': 2}),
                                    dbc.Col([drop_down],
                                            width={'size': 6}),
                                    dbc.Col(
                                        html.Img(src=logo_fpath,
                                                 height='40px',
                                                 className=CSS['logo']),
                                        width={'size': 2, 'offset': 2}),
                                      ],
                                     className=CSS['file-row'],
                                     align='center',
                                     )
        # Layout for EEG/ICA and Topo plots row
        timeseries_div = html.Div([self.eeg_visualizer.container_plot,
                                   self.ica_visualizer.container_plot],
                                  id='channel-and-icsources-div',
                                  className=CSS['timeseries-col'])
        visualizers_row = dbc.Row([dbc.Col([timeseries_div], width=8),
                                   dbc.Col(self.ica_topo.container_plot,
                                           className=CSS['topo-col'],
                                           width=4)],
                                  style=STYLE['plots-row'],
                                  className=CSS['plots-row']
                                  )
        # Final Layout
        qc_app_layout = dbc.Container([control_header_row, visualizers_row],
                                      fluid=True, style=STYLE['qc-container'])
        self.app.layout.children.append(qc_app_layout)

    def load_recording(self, fname):
        """  """
        fname = Path(fname)
        ica_fpath = fname.parent / fname.name.replace("_eeg.edf", "_ica1_ica.fif") 
        iclabel_fpath = fname.parent / fname.name.replace("_eeg.edf", "_iclabels.tsv") 
        self.raw, self.raw_ica, self.ica = \
            get_ll_assets(fname, ica_fpath, iclabel_fpath)[:-1]

        self.ic_types = pd.read_csv(iclabel_fpath, sep='\t')
        self.ic_types['component'] = [f'ICA{ic:03}'
                                        for ic in self.ic_types.component]
        self.ic_types = self.ic_types.set_index('component')['ic_type']
        self.ic_types = self.ic_types.to_dict()
        cmap = {ic: ic_label_cmap[ic_type]
                for ic, ic_type in self.ic_types.items()}
        self.ica_visualizer.load_recording(self.raw_ica, cmap=cmap,
                                           ic_types=self.ic_types)
        self.eeg_visualizer.load_recording(self.raw)
        self.ica_topo.load_recording(self.raw.get_montage(), self.ica, self.ic_types)    

    def set_callbacks(self):
        @self.app.callback(
            Output('file-dropdown', 'options'),
            Input('folder-selector', 'n_clicks'),
        )
        def folder_button_clicked(n_clicks):
            if n_clicks:
                root = tkinter.Tk()
                root.withdraw()
                directory = Path(filedialog.askdirectory())
                print('selected directory: ', directory)
                # self.eeg_visualizer.change_dir(directory)
                '''files_list = [dbc.DropdownMenuItem(str(file.name), id=str(file))
                              for file in sorted(self.project_root.rglob("*.edf"))]'''
                files_list = [{'label': str(file.name), 'value': str(file)}
                              for file
                              in sorted(self.project_root.rglob("*.edf"))]
                root.destroy()
                print('***', files_list)
                return files_list # directory
            return dash.no_update


        @self.app.callback(
            Output('file-dropdown', 'placeholder'),
            Input('file-dropdown', 'value')
        )
        def file_selected(value):
            if value:  # on selection of dropdown item
                self.load_recording(value)
            return dash.no_update
