"""Command Line Interface entry to launch Lossless QC Dash app."""
# Authors: Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: MIT
import subprocess
import argparse
from pylossless.dash.app import get_app

desc = 'Launch QCR dashboard with optional directory and filename arguments.'


def launch_dash_app(directory=None, filepath=None, disable_buttons=False,
                    debug=False, host='127.0.0.1', port=None, timeout=30,
                    deployment=False):
    """Launch dashboard."""
    if not port:
        port = 8050
    if not host:
        host = '127.0.0.1'
    if not timeout:
        timeout = 30
    app = get_app(fpath=filepath, project_root=directory,
                  disable_buttons=disable_buttons)
    if not deployment:
        app.run_server(debug=debug, host=host, port=port)
    else:
        # Call gunicorn to launch your Dash app
        log = 'debug' if debug else 'INFO'  # '--chdir', './pylossless/dash',
        cmd = ['gunicorn', 'pylossless.dash.app:server',
               '-b', f"{host}:{port}", f'--log-level={log}',
               '-t', f"{timeout}",
               '--env', f'fpath={filepath}',
               '--env', f'disable_buttons={disable_buttons}']
        subprocess.Popen(cmd)


def main(deployment=False):
    """Parse arguments for CLI."""
    disable_button_help = ('If included, Folder and Save buttons are'
                           ' deactivated')
    port_help = ('port used to serve application. If not set, defaults to 8050'
                 )
    timeout_help = {'set the timeout in gunicorn for the deployment server.'
                    ' default is 30 seconds'}
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', help='path to the project folder')
    parser.add_argument('--filepath', help='path to the EDF file to load')
    parser.add_argument('--disable_buttons', action='store_true',
                        help=disable_button_help)
    parser.add_argument('--debug', action='store_true',
                        help='If set, run application in debug mode')
    parser.add_argument('--host', help='host IP used to serve application')
    parser.add_argument('--port', help=port_help)
    parser.add_argument('--deployment', action='store_true', help="launch in deployment server.")
    parser.add_argument('--timeout', help=timeout_help)
    args = parser.parse_args()
    launch_dash_app(args.directory, args.filepath, args.disable_buttons,
                    args.debug, args.host, args.port,
                    deployment=args.deployment, timeout=args.timeout)


if __name__ == '__main__':
    main()
