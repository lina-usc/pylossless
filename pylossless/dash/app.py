from pathlib import Path
import dash

import dash_bootstrap_components as dbc

from pylossless.dash.qcgui import QCGUI, get_ll_assets


def get_app(assets_kwargs=None, kind="dash"):

    if assets_kwargs is None:
        assets_kwargs = {}
    if kind == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)

    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    QCGUI(app, *get_ll_assets(**assets_kwargs))
    return app    



if __name__ == '__main__':

    '''assets_kwargs = dict(fname=Path(__file__).parent / '../../assets/test_data/sub-s02/eeg/sub-s02_task-faceO_eeg.edf',
                         ica_fpath=Path(__file__).parent / '../../assets/test_data/sub-s02/eeg/sub-s02_task-faceO_ica2_ica.fif',
                         iclabel_fpath = Path(__file__).parent / '../../assets/test_data/sub-s02/eeg/sub-s02_task-faceO_iclabels.tsv',
                         bads=('A27', 'A5', 'B10', 'B16', 'B17', 'B27', 'B28',
                               'C10', 'C17','C18', 'C3', 'D1'))'''
    assets_kwargs = dict(fname=None,
                         ica_fpath=None,
                         iclabel_fpath=None,
                         bads=('A27', 'A5', 'B10', 'B16', 'B17', 'B27', 'B28',
                               'C10', 'C17','C18', 'C3', 'D1'))
    get_app(assets_kwargs).run_server(debug=True, use_reloader=False)

"""
from werkzeug.middleware.profiler import ProfilerMiddleware
if __name__ == "__main__":
    import os

    test_fpath = Path(__file__).parent / '../../assets/test_data'
    eeg_fpath = 'sub-s02/eeg/sub-s02_task-faceO_eeg.edf'
    ica_fpath = 'sub-s02/eeg/sub-s02_task-faceO_ica2_ica.fif'
    iclabel_fpath = 'sub-s02/eeg/sub-s02_task-faceO_iclabels.tsv'
    assets_kwargs = dict(fname=test_fpath / eeg_fpath,
                         ica_fpath=test_fpath / ica_fpath,
                         iclabel_fpath=test_fpath / iclabel_fpath,
                         bads=('A27', 'A5', 'B10', 'B16', 'B17', 'B27', 'B28',
                               'C10', 'C17','C18', 'C3', 'D1'))
    app = get_app(assets_kwargs)
    if False: # os.getenv("PROFILER", None):
        app.server.config["PROFILE"] = True
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app, profile_dir='.', stream=None, sort_by=("cumtime", "tottime"), restrictions=[50]
        )
    app.run_server(debug=False)
"""