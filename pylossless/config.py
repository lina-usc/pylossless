import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "ll_default_config.json"


def save_json(init_variables, file_name='init_variables.json'):
    with open(file_name, "w") as init_variables_file:
        json.dump(init_variables, init_variables_file,
                  indent=4, sort_keys=True)


def read_json(file_name):
    with open(file_name, "r") as init_variables_file:
        return json.load(init_variables_file)


def create_init_variables_json(path_in, id_, run, session):
    init_variables = {'path_in': path_in,
                      'id': id_,
                      'run': run,
                      'session': session
                      }
    save_json(init_variables)


def get_default_config():
    return read_json(DEFAULT_CONFIG_PATH)