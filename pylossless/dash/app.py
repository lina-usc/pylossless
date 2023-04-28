# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Launching point for Lossless QC Dash app."""
import dash
import dash_bootstrap_components as dbc
from pylossless.dash.qcgui import QCGUI


def get_app(fpath=None, project_root=None, kind="dash"):
    """Wrapper function for calling either Dash or Jupyter
    for Lossless QC procedure."""

    if kind == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SLATE])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    QCGUI(app, fpath=fpath, project_root=project_root)
    return app


if __name__ == '__main__':
    get_app().run_server(debug=True, use_reloader=False)
