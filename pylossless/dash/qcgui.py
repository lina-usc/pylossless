from dash import dcc, html
from dash.dependencies import Input, Output

# file selection
import tkinter
from tkinter import filedialog

from pathlib import Path

from .topo_viz import TopoVizICA
from .mne_visualizer import MNEVisualizer


class QCGUI:

    def __init__(self, app, raw, raw_ica, ica, project_root='./tmp_test_files'):

        self.project_root = Path(project_root)

        self.app = app
        self.raw = raw
        self.raw_ica = raw_ica
        self.ica = ica

        self.ica_visualizer = None
        self.eeg_visualizer = None
        self.ica_topo = None

        self.set_layout()
        self.set_callbacks()


    def annot_created_callback(self, annotation):
        self.raw.set_annotations(self.raw.annotations + annotation)
        self.raw_ica.set_annotations(self.raw_ica.annotations + annotation)
        self.ica_visualizer.update_layout(ch_slider_val=self.ica_visualizer.channel_slider.max,
                                    time_slider_val=self.ica_visualizer.win_start)
        self.eeg_visualizer.update_layout()

    def set_visualizers(self):
        # Setting time-series and topomap visualizers
        self.ica_visualizer = MNEVisualizer(self.app, self.raw_ica, dash_id_suffix='ica', annot_created_callback=self.annot_created_callback)
        self.eeg_visualizer = MNEVisualizer(self.app, self.raw, time_slider=self.ica_visualizer.dash_ids['time-slider'], 
                                    dcc_graph_kwargs=dict(config={'modeBarButtonsToRemove':['zoom','pan']}),
                                    annot_created_callback=self.annot_created_callback)
        self.ica_topo = TopoVizICA(self.app, self.raw.get_montage(), self.ica, 
                                   topo_slider_id=self.ica_visualizer.dash_ids['ch-slider'])

        self.ica_visualizer.new_annot_desc = 'bad_manual'
        self.eeg_visualizer.new_annot_desc = 'bad_manual'

        self.ica_visualizer.update_layout()        

    def set_layout(self):
        # app.layout must not be None for some of the operations of the visualizers.
        self.app.layout = html.Div([])
        self.set_visualizers()

        derivatives_dir = self.project_root / 'derivatives'
        files_list = [{'label':str(file), 'value':str(file)} for file in sorted(derivatives_dir.rglob("*.edf"))]

        control_header_div = html.Div([
                                        html.Button('Folder',
                                                    id='submit-val',
                                                    className="folderButton",
                                                    title=f'current folder: {self.project_root.resolve()}'
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
                                    )
        visualizers_div = html.Div([
                                    html.Div(id='plots-container', 
                                            children=[html.Div([self.eeg_visualizer.container_plot,
                                                                self.ica_visualizer.container_plot]),
                                                    self.ica_topo.container_plot,
                                                    ],
                                            style={"border":"2px green solid"})
                                        ],
                                    style={'display':'block'})

        qc_app_layout = html.Div([control_header_div, visualizers_div], style={"display":"block"})
        self.app.layout.children.append(qc_app_layout)

    def set_callbacks(self):
        @self.app.callback(
            Output('container-button-basic', 'children'),
            Input('submit-val', 'n_clicks')
        )
        def _select_folder(n_clicks):
            if n_clicks:
                root = tkinter.Tk()
                root.withdraw()
                directory = filedialog.askdirectory()
                print('selected directory: ', directory)
                root.destroy()
                self.eeg_visualizer.change_dir(directory)
                return directory


