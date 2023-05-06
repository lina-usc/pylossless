"""Command Line Interface entry to launch Lossless QC Dash app."""
# Authors: Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: MIT

import argparse
from pylossless.dash.app import get_app

desc = 'Launch QCR dashboard with optional directory and filename arguments.'


def launch_dash_app(directory=None, filepath=None, disable_buttons=False,
                    debug=False, host='127.0.0.1', port=8050):
    """Launch dashboard."""
    app = get_app(fpath=filepath, project_root=directory,
                  disable_buttons=disable_buttons)
    app.run_server(debug=debug, host=host, port=port)


def main():
    """Parse arguments for CLI."""
    disable_button_help = ('If included, Folder and Save buttons are'
                           ' deactivated')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', help='path to the project folder')
    parser.add_argument('--filepath', help='path to the EDF file to load')
    parser.add_argument('--disable_buttons', action='store_true',
                        help=disable_button_help)
    parser.add_argument('debug', action='store_True',
                        help='If set, run application in debug mode')
    parser.add_argument('--host', help='host IP used to serve application')
    parser.add_argument('--port', help='port used to serve application')
    args = parser.parse_args()
    launch_dash_app(args.directory, args.filepath, args.disable_buttons,
                    args.debug, args.host, args.port)


if __name__ == '__main__':
    main()
