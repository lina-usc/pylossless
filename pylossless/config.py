import yaml
from pathlib import Path
import sys


class Config(dict):

    DEFAULT_CONFIG_PATH = (Path(__file__).parent.parent /
                        "assets" / "ll_default_config.yaml")

    def __init__(self, file_name=None):
        self.file_name = Path(file_name)

    def read(self, file_name):
        self.file_name = Path(file_name)

        with self.file_name.open("r") as init_variables_file:
            self.update(yaml.safe_load(init_variables_file))

    def load_default(self):
        file_name_bck = self.file_name
        self.read(Config.DEFAULT_CONFIG_PATH)

        # Restauring the file_name. We don't want this attribute
        # to point to the default config, as it make it liekly
        # that the user will end up overwritting the defaut config.
        self.file_name = file_name_bck

    def save(self, file_name=None):

        if file_name is not None:
            self.file_name = Path(file_name)
        else:
            if self.file_name is None:
                self.file_name = Path('init_variables.yaml')

        with self.file_name.open("w") as init_variables_file:
            yaml.dump(dict(self), init_variables_file,
                    indent=4, sort_keys=True)

    def print(self):
        yaml.dump(dict(self), sys.stdout, indent=4, sort_keys=True)
