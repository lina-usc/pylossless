import yaml
from pathlib import Path


DEFAULT_CONFIG_PATH = (Path(__file__).parent.parent / 
                       "assets" / "ll_default_config.yaml")


def save_config(init_variables=None, path_in=None, id_=None, run=None, 
                session=None, file_name='init_variables.yaml'):

    if init_variables is None:
        init_variables = {}
    if path_in is not None:
        init_variables.update({"path_in": path_in})
    if id_ is not None:
        init_variables.update({"id": id_})
    if run is not None:
        init_variables.update({"run": run})
    if session is not None:
        init_variables.update({"session": session})
    
    with open(file_name, "w") as init_variables_file:
        yaml.dump(init_variables, init_variables_file,
                  indent=4, sort_keys=True)

def read_config(file_name):
    with open(file_name, "r") as init_variables_file:
        return yaml.safe_load(init_variables_file)

def get_default_config():
    return read_config(DEFAULT_CONFIG_PATH)
