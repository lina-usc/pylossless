from pathlib import Path

# file selection
import tkinter
from tkinter import filedialog
import numpy as np

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# loading raw object
from mne_bids import get_bids_path_from_fname
import mne

from .topo_viz import TopoVizICA
from .mne_visualizer import MNEVisualizer, ICVisualizer

from . import ic_label_cmap
from ..pipeline import LosslessPipeline

from .css_defaults import CSS, STYLE


class QCGUI:

    def __init__(self, app, fpath=None, project_root=None, verbose=False):

        # TODO: Fix this pathing indexing, can likely cause errors.
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
            project_root = project_root / 'assets' / 'test_data'
        self.project_root = Path(project_root)

        self.pipeline = LosslessPipeline()
        self.app = app
        self.raw = None
        self.ica = None
        self.raw_ica = None
        self.fpath = None
        if fpath:
            self.load_recording(fpath)

        self.ica_visualizer = None
        self.eeg_visualizer = None
        self.ica_topo = None
        self.ic_types = None
        self.set_layout()
        self.set_callbacks()

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

        # Using the ouptut of the callback being triggered by
        # a selection of a new file, so that the two callbacks
        # are executed sequentially.
        refresh_inputs = [Input('file-dropdown', 'placeholder')]
        self.ica_visualizer = ICVisualizer(
            self.app, self.raw_ica,
            dash_id_suffix='ica',
            annot_created_callback=self.annot_created_callback,
            cmap=cmap,
            ic_types=self.ic_types,
            refresh_inputs=refresh_inputs, 
            set_callbacks=False)

        self.eeg_visualizer = MNEVisualizer(
            self.app,
            self.raw,
            annot_created_callback=self.annot_created_callback,
            refresh_inputs=refresh_inputs,
            show_time_slider=True,
            set_callbacks=False)
        
        input_ = Input(self.eeg_visualizer.dash_ids['graph'], "relayoutData")
        self.ica_visualizer.refresh_inputs.append(input_)
        input_ = Input(self.ica_visualizer.dash_ids['graph'], "relayoutData")
        self.eeg_visualizer.refresh_inputs.append(input_)
        self.eeg_visualizer.set_callback()
        self.ica_visualizer.set_callback()


        self.ica_visualizer.mne_annots = self.eeg_visualizer.mne_annots
        self.ica_visualizer.dash_ids['mne-annotations'] = self.eeg_visualizer.dash_ids['mne-annotations']

        montage = self.raw.get_montage() if self.raw else None
        self.ica_topo = TopoVizICA(self.app, montage, self.ica, self.ic_types,
                                   show_sensors=True,
                                   refresh_inputs=refresh_inputs)

        self.ica_visualizer.new_annot_desc = 'bad_manual'
        self.eeg_visualizer.new_annot_desc = 'bad_manual'

        self.ica_visualizer.update_layout()

    def update_bad_ics(self):
        df = self.pipeline.flagged_ics.data_frame
        ic_names = self.raw_ica.ch_names
        df['ic_names'] = ic_names
        df.set_index('ic_names', inplace=True)
        for ic_name in self.raw_ica.ch_names:
            ic_type = df.loc[ic_name, 'ic_type']
            is_mne_bad = ic_name in self.raw_ica.info['bads']
            was_mne_bad = df.loc[ic_name, 'annotate_method'] == 'manual'
            if is_mne_bad:  # user added channel to info['bads']
                df.loc[ic_name, 'annotate_method'] = 'manual'
                df.loc[ic_name, 'status'] = 'bad'
            elif was_mne_bad and not is_mne_bad:
                df.loc[ic_name, 'annotate_method'] = np.nan
                if ic_type == "brain":
                    df.loc[ic_name, 'status'] = 'good'
        df.reset_index(drop=True, inplace=True)
        # TODO understand why original component values are lost to begin with
        df["component"] = np.arange(df.shape[0])

    def set_layout(self):
        # app.layout must not be None for some of the operations of the
        # visualizers.
        self.app.layout = html.Div([])
        self.set_visualizers()

        # Layout for file control row
        # derivatives_dir = self.project_root / 'derivatives'
        files_list = []
        files_list = [{'label': str(file.name), 'value': str(file)}
                      for file
                      in sorted(self.project_root.rglob("*.edf"))]

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
        self.drop_down = dcc.Dropdown(id='file-dropdown',
                                      className=CSS['dropdown'],
                                      placeholder="Select a file",
                                      options=files_list)
        control_header_row = dbc.Row([
                                    dbc.Col([folder_button, save_button],
                                            width={'size': 2}),
                                    dbc.Col([self.drop_down,
                                             html.P(id='dropdown-output')],
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

    def load_recording(self, fpath, verbose=False):
        """  """
        self.fpath = Path(fpath)
        # iclabel_fpath = self.fpath.parent /
        #   self.fpath.name.replace("_eeg.edf", "_iclabels.tsv")
        self.pipeline.load_ll_derivative(self.fpath)
        self.raw = self.pipeline.raw
        self.ica = self.pipeline.ica2
        if self.raw:
            info = mne.create_info(self.ica._ica_names,
                                   sfreq=self.raw.info['sfreq'],
                                   ch_types=['eeg'] * self.ica.n_components_,
                                   verbose=verbose)
            sources = self.ica.get_sources(self.raw).get_data()
            self.raw_ica = mne.io.RawArray(sources, info, verbose=verbose)
            self.raw_ica.set_meas_date(self.raw.info['meas_date'])
            self.raw_ica.set_annotations(self.raw.annotations)
            df = self.pipeline.flagged_ics.data_frame
            ic_names = self.raw_ica.ch_names
            df['ic_names'] = ic_names

            bads = [ic_name
                    for ic_name, annot
                    in df[["ic_names", "annotate_method"]].values
                    if annot == "manual"]
            self.raw_ica.info["bads"] = bads
        else:
            self.raw_ica = None

        # pd.read_csv(iclabel_fpath, sep='\t')
        self.ic_types = self.pipeline.flagged_ics.data_frame
        self.ic_types['component'] = [f'ICA{ic:03}'
                                      for ic in self.ic_types.component]
        self.ic_types = self.ic_types.set_index('component')['ic_type']
        self.ic_types = self.ic_types.to_dict()
        cmap = {ic: ic_label_cmap[ic_type]
                for ic, ic_type in self.ic_types.items()}
        self.ica_visualizer.load_recording(self.raw_ica, cmap=cmap,
                                           ic_types=self.ic_types)
        self.eeg_visualizer.load_recording(self.raw)
        self.ica_topo.load_recording(self.raw.get_montage(),
                                     self.ica, self.ic_types)

    def set_callbacks(self):
        """@self.app.callback(
            Output('file-dropdown', 'options'),
            Input('folder-selector', 'n_clicks'),
        )
        def folder_button_clicked(n_clicks):
            if n_clicks:
                root = tkinter.Tk()
                # root.focus_force()
                # Make folder picker dialog appear on top of other windows
                root.wm_attributes('-topmost', 1)
                # Cause the root window to disappear milliseconds
                # after calling the filedialog.
                root.after(100, root.withdraw)
                directory = Path(filedialog.askdirectory())
                print('selected directory: ', directory)
                # self.eeg_visualizer.change_dir(directory)
                '''files_list = [dbc.DropdownMenuItem(str(f.name), id=str(f))
                    for f in sorted(self.project_root.rglob("*.edf"))]'''
                files_list = [{'label': str(file.name), 'value': str(file)}
                              for file
                              in sorted(self.project_root.rglob("*.edf"))]
                root.destroy()
                print('***', files_list)
                return files_list  # directory
            return dash.no_update"""

        @self.app.callback(
            Output('file-dropdown', 'placeholder'),
            Input('file-dropdown', 'value'),
            prevent_initial_call=True
        )
        def file_selected(value):
            print('IN THE FILE CALLBACK')
            if value:  # on selection of dropdown item
                self.load_recording(value)
                return value

        @self.app.callback(
            Output('dropdown-output', 'children'),
            Input('save-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def save_file(n_clicks):
            self.update_bad_ics()
            self.pipeline.save(get_bids_path_from_fname(self.fpath),
                               overwrite=True)
            print('file saved!')
            return dash.no_update

        @self.app.callback(
                Output(self.eeg_visualizer.dash_ids['time-slider'],
                       'value'),
                Output(self.ica_visualizer.dash_ids['time-slider'],
                       'value'),
                Output(self.eeg_visualizer.dash_ids['time-slider'],
                       component_property='min'),
                Output(self.eeg_visualizer.dash_ids['time-slider'],
                       component_property='max'),
                Output(self.ica_visualizer.dash_ids['time-slider'],
                       component_property='min'),
                Output(self.ica_visualizer.dash_ids['time-slider'],
                       component_property='max'),
                Input(self.eeg_visualizer.dash_ids['time-slider'],
                      'value'),
                Input(self.ica_visualizer.dash_ids['time-slider'],
                      'value'),
                Input('file-dropdown', 'placeholder'),
                prevent_initial_call=True)
        def sync_time_sliders(eeg_time_slider, ica_time_slider, selected_file):
            ctx = dash.callback_context
            print('IN THE TIME CALLBACK')
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == self.eeg_visualizer.dash_ids['time-slider']:
                    value = eeg_time_slider
                    return no_update, value, no_update, no_update, no_update, no_update
                if trigger_id == self.ica_visualizer.dash_ids['time-slider']:
                    value = ica_time_slider
                    return value, no_update, no_update, no_update, no_update, no_update
                if trigger_id == 'file-dropdown':
                    print('HITTTT')
                    value = self.ica_visualizer.time_slider.value
                    min_ = self.ica_visualizer.time_slider.min
                    max_ = self.ica_visualizer.time_slider.max
                    print('#### ', value)
                    return value, value, min_, max_, min_, max_

        @self.app.callback(
                Output(self.ica_visualizer.dash_ids['ch-slider'],
                       'value'),
                Output('topo-slider',  'value'),
                Input(self.ica_visualizer.dash_ids['ch-slider'],
                      'value'),
                Input('topo-slider', 'value'),
                prevent_initial_call=True)
        def sync_ica_sliders(ica_ch_slider, ica_topo_slider):
            ctx = dash.callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == self.ica_visualizer.dash_ids['ch-slider']:
                    value = ica_ch_slider
                    return no_update, value
                elif trigger_id == 'topo-slider':
                    value = ica_topo_slider
                    return value, no_update
