"""Command Line Interface entry to launch Lossless QC Dash app."""
# Authors: Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: MIT

import webbrowser
from threading import Timer
import argparse
from pylossless.dash.app import get_app

desc = "Launch QCR dashboard with optional directory and filename arguments."


def open_browser(port=8050):
    """Open a webbrowser."""
    webbrowser.open_new("http://localhost:{}".format(port))


def launch_dash_app(directory=None, filepath=None, disable_buttons=False):
    """Launch dashboard."""
    app = get_app(
        fpath=filepath, project_root=directory, disable_buttons=disable_buttons
    )

    Timer(1, open_browser).start()
    # We can use either (debug=False) or (debug=True, use_reloader=False)
    # if we don't want the code to open the browser twice.
    # Explanations: https://stackoverflow.com/a/68686551/1825043
    app.run_server(debug=True, use_reloader=False)


def main():
    """Parse arguments for CLI."""
    disable_button_help = "If included, Folder and Save buttons are" " deactivated"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--directory", help="path to the project folder")
    parser.add_argument("--filepath", help="path to the EDF file to load")
    parser.add_argument(
        "--disable_buttons", action="store_true", help=disable_button_help
    )
    args = parser.parse_args()
    launch_dash_app(args.directory, args.filepath, args.disable_buttons)


if __name__ == "__main__":
    main()
