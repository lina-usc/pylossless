# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Python file to instantiate the graphs for the qcr app."""

from pathlib import Path

# file selection
import tkinter as tk
from tkinter import filedialog
import multiprocessing

import numpy as np

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# loading raw object
from mne_bids import get_bids_path_from_fname
import mne
from mne.utils import logger

from .topo_viz import TopoVizICA
from .mne_visualizer import MNEVisualizer, ICVisualizer

from . import ic_label_cmap
from ..pipeline import LosslessPipeline

from .css_defaults import CSS, STYLE


def open_folder_dialog():
    """Provide user with a finder window to select a directory."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory()
    root.destroy()
    return file_path


class QCGUI:
    """Class that stores the visualizer-plots that are used in the qcr app."""

    def __init__(self, app,
                 fpath=None, project_root=None,
                 disable_buttons=False,
                 verbose=False):
        """Initialize class.

        Parameters
        ----------
        app : dash.app
            The dash.app object that will host the graphs. This is usually
            defined in a python file named app.py
        fpath : mne_bids.BIDSPath | pathlib.Path | None
            The EEG file to read. Should be the file in one of the subject
            folders in a derivatives/pylossless directory that was created
            by the output of the pylossless pipeline. If a pathlib.Path, will
            try to convert to mne_bids.BIDSPath using get_bids_path_from_fname.
        project_root : pathlib.Path | None
            Should be the path to a derivatives/pylossless directory.
        verbose : None
            The verbosity level to set for MNE/mne_bids/pylossless functions.
        """
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
        self.ica_visualizer = None
        self.eeg_visualizer = None
        self.ica_topo = None
        self.ic_types = None
        self.set_layout(disable_buttons=disable_buttons)
        self.set_callbacks()
        if fpath:
            self.fpath = Path(fpath)
            # self.load_recording(fpath)

    def set_visualizers(self):
        """Create EEG/ICA time-series dcc.graphs and topomap dcc.graphs."""
        # Setting time-series and topomap visualizers
        if self.ic_types:
            cmap = {ic: ic_label_cmap[ic_type]
                    for ic, ic_type in self.ic_types.items()}
        else:
            cmap = None

        # Using the output of the callback being triggered by
        # a selection of a new file, so that the two callbacks
        # are executed sequentially.
        refresh_inputs = [Input('file-dropdown', 'placeholder')]
        self.ica_visualizer = ICVisualizer(
            self.app, self.raw_ica,
            dash_id_suffix='ica',
            cmap=cmap,
            ic_types=self.ic_types,
            refresh_inputs=refresh_inputs,
            set_callbacks=False)

        self.eeg_visualizer = MNEVisualizer(
            self.app,
            self.raw,
            refresh_inputs=refresh_inputs.copy(),
            show_time_slider=False,
            set_callbacks=False)

        input_ = Input(self.eeg_visualizer.dash_ids['graph'], "relayoutData")
        self.ica_visualizer.refresh_inputs.append(input_)
        input_ = Input(self.ica_visualizer.dash_ids['graph'], "relayoutData")
        self.eeg_visualizer.refresh_inputs.append(input_)

        self.ica_visualizer.set_callback()
        self.eeg_visualizer.set_callback()

        self.ica_visualizer.mne_annots = self.eeg_visualizer.mne_annots
        self.ica_visualizer.dash_ids['mne-annotations'] = \
            self.eeg_visualizer.dash_ids['mne-annotations']

        montage = self.raw.get_montage() if self.raw else None
        self.ica_topo = TopoVizICA(self.app, montage, self.ica, self.ic_types,
                                   show_sensors=True,
                                   refresh_inputs=refresh_inputs)

        self.ica_visualizer.new_annot_desc = 'bad_manual'
        self.eeg_visualizer.new_annot_desc = 'bad_manual'

        self.ica_visualizer.update_layout()

    def update_bad_ics(self):
        """Add IC name to raw.info['bads'] after selection by user in app."""
        df = self.pipeline.flags['ic'].data_frame
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

    def set_layout(self, disable_buttons=False):
        """Create the app.layout for the app object.

        Notes
        -----
        This specifies the layout for all the dash components in the
        qcr dashboard. I.e. it specifies the navbar at the top, the placement
        of the timeseries graphs and topoplot graphs, etc.
        """
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
                                   title=dropdown_text,
                                   disabled=disable_buttons)
        save_button = dbc.Button('Save', id='save-button',
                                 color='info',
                                 outline=True,
                                 className=CSS['button'],
                                 disabled=disable_buttons)
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
        """Load the EEG/ICA raw recording from file.

        Parameters
        ----------
        fpath : mne_bids.BIDSPath | pathlib.Path
            Should contain a path to a subject folder in the
            derivatives/pylossless directory that was created by the
            pylossless pipeline.
        """
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
            df = self.pipeline.flags['ic'].data_frame
            ic_names = self.raw_ica.ch_names
            df['ic_names'] = ic_names

            bads = [ic_name
                    for ic_name, annot
                    in df[["ic_names", "annotate_method"]].values
                    if annot == "manual"]
            self.raw_ica.info["bads"] = bads
        else:
            self.raw_ica = None

        self.ic_types = self.pipeline.flags['ic'].data_frame
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
        """Define additional callbacks that will be used by the qcr app."""
        # TODO: delete this folder selection callback
        @self.app.callback(
            Output('file-dropdown', 'options'),
            Input('folder-selector', 'n_clicks'),
            prevent_initial_call=True
        )
        def folder_button_clicked(n_clicks):
            if n_clicks:
                with multiprocessing.Pool(processes=1) as pool:
                    folder_path = pool.apply(open_folder_dialog)
                    self.project_root = Path(folder_path)

                files_list = [{'label': str(file.name), 'value': str(file)}
                              for file
                              in sorted(self.project_root.rglob("*.edf"))]
                return files_list
            return dash.no_update

        @self.app.callback(
            Output('file-dropdown', 'placeholder'),
            Input('file-dropdown', 'value'),
            prevent_initial_call=False
        )
        def file_selected(value):
            if value:  # on selection of dropdown item
                self.load_recording(value)
                return value
            # Needed when an initial fpath is set from the CLI
            if self.fpath:
                self.load_recording(self.fpath)
                return str(self.fpath.name)

        @self.app.callback(
            Output('dropdown-output', 'children'),
            Input('save-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def save_file(n_clicks):
            self.update_bad_ics()
            self.eeg_visualizer.update_inst_annnotations()
            self.pipeline.save(get_bids_path_from_fname(self.fpath),
                               overwrite=True)
            logger.info('file saved!')
            return dash.no_update

        properties = ["value", "min", "max", "marks"]
        slider_ids = [self.eeg_visualizer.dash_ids['time-slider'],
                      self.ica_visualizer.dash_ids['time-slider']]
        sliders = [self.ica_visualizer.time_slider,
                   self.eeg_visualizer.time_slider]
        decorator_args = []
        for slider_id in slider_ids:
            decorator_args += [Output(slider_id, property)
                               for property in properties]
        for slider_id in slider_ids:
            decorator_args += [Input(slider_id, 'value')]
        decorator_args += [Input('file-dropdown', 'placeholder')]

        @self.app.callback(*decorator_args, prevent_initial_call=True)
        def sync_time_sliders(*args):
            """Sync EEG and ICA-Raw time sliders and refresh upon new file.

            Parameters
            ----------
            eeg_time_slider : dcc.Slider
               The MNEVisualizer.time_slider component property for EEG.
            ica_time_slider : dcc.Slider
                The MNEVisualizer.time_slider component property for ICA Raw.
            selected file : dcc.Dropdown
                The file-dropdown dash component. The placeholder component
                property of this dash-component is used to refresh the time
                sliders when loading a new file.
            Returns
            -------
            For the following component_properties of the EEG time slider and
            ICA raw time slider: value, min, max, marks: Either the new values
            for these components or dash.no_update to prevent updating the
            component property. The returns are in this order:

            1. eeg_visualizer.time_slider.value
            2. ica_visuazlier.time_slider.value
            3. eeg_visualizer.time_slider.min
            4. ica_visualizer.time_slider.min
            5. eeg_visualizer.time_slider.max
            6. ica_visuailzer.time_slider.max
            7. eeg_visualizer.time_slider.marks
            8. ica_visualizer.time_slider.marks

            Notes
            -----
            This callback refreshes slider on new file selection and when
            the slider of the other time-series graph is changed.
            """
            ctx = dash.callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == slider_ids[0]:
                    return ([no_update]*len(properties) + [args[0]] +
                            [no_update]*(len(properties)-1))
                if trigger_id == slider_ids[1]:
                    return [args[1]] + [no_update]*(len(properties)*2-1)
                if trigger_id == 'file-dropdown':
                    args = []
                    for slider in sliders[::-1]:
                        args += [getattr(slider, property)
                                 for property in properties]
                    return args

        @self.app.callback(
                Output(self.ica_visualizer.dash_ids['ch-slider'],
                       'value'),
                Output('topo-slider',  'value'),
                Output(self.ica_visualizer.dash_ids['ch-slider'],
                       component_property='min'),
                Output(self.ica_visualizer.dash_ids['ch-slider'],
                       component_property='max'),
                Output('topo-slider',
                       component_property='min'),
                Output('topo-slider',
                       component_property='max'),
                Input(self.ica_visualizer.dash_ids['ch-slider'],
                      'value'),
                Input('topo-slider', 'value'),
                Input('file-dropdown', 'placeholder'),
                prevent_initial_call=True)
        def sync_ica_sliders(ica_ch_slider, ica_topo_slider, selected_file):
            """Sync ICA-Raw and ICA-topo sliders and refresh upon new file."""
            ctx = dash.callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == self.ica_visualizer.dash_ids['ch-slider']:
                    # If user dragged the ica-raw ch-slider
                    value = ica_ch_slider
                    # only update the ica-topo slider val.
                    return (no_update, value,
                            no_update, no_update,  # min max 4 ica-raw slider.
                            no_update, no_update)  # min max 4 topo-slider
                if trigger_id == 'topo-slider':
                    # If the user dragged the topoplot slider
                    value = ica_topo_slider
                    # only update the ica-raw ch-slider val
                    return (value, no_update,
                            no_update, no_update,  # min max 4 ica-raw slider
                            no_update, no_update)  # min max 4 topo-slider
                if trigger_id == 'file-dropdown':
                    # If the user selected a new file
                    value = self.ica_visualizer.channel_slider.value
                    min_ = self.ica_visualizer.channel_slider.min
                    max_ = self.ica_visualizer.channel_slider.max
                    # update the val, min, max component-properties for both
                    # the ica-raw ch-slider and topoplot slider
                    return value, value, min_, max_, min_, max_

        @self.app.callback(
                Output(self.eeg_visualizer.dash_ids['ch-slider'],
                       'value'),
                Output(self.eeg_visualizer.dash_ids['ch-slider'],
                       component_property='min'),
                Output(self.eeg_visualizer.dash_ids['ch-slider'],
                       component_property='max'),
                Input('file-dropdown', 'placeholder'),
                prevent_initial_call=True)
        def refresh_eeg_ch_slider(selected_file):
            """Refresh eeg graph ch-slider upon new file selection."""
            ctx = dash.callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == 'file-dropdown':
                    # If the user selected a new file
                    value = self.eeg_visualizer.channel_slider.value
                    min_ = self.eeg_visualizer.channel_slider.min
                    max_ = self.eeg_visualizer.channel_slider.max
                    # update the val, min, max component-properties for
                    # the  eeg ch-slider
                    return value, min_, max_
