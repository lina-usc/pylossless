# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Launching point for Lossless QC Dash app."""
import os
import dash
import dash_bootstrap_components as dbc
from pylossless.dash.qcgui import QCGUI


def get_app(fpath=None, project_root=None, disable_buttons=False, kind="dash",
            deployment=False):
    """Call either Dash or Jupyter for Lossless QC procedure."""
    if kind == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SLATE])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    if deployment:
        fpath = os.environ.get("fpath")
        print(f'##### {fpath}')
        disable_buttons = os.environ.get("disable_buttons")
        print(f'$$$$$ {disable_buttons}')
    QCGUI(app, fpath=fpath, project_root=project_root,
          disable_buttons=disable_buttons)
    return app


app = get_app(deployment=True)
server = app.server  # Make server object available for deployment servers
if __name__ == '__main__':  # For running development server locally
    app.run_server(debug=True, use_reloader=False)
