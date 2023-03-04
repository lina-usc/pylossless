import dash

import dash_bootstrap_components as dbc

from pylossless.dash.qcgui import QCGUI


def get_app(fpath=None, kind="dash"):
    """ """
    if kind == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    QCGUI(app, fpath=fpath)
    return app


if __name__ == '__main__':
    get_app().run_server(debug=True, use_reloader=False)
