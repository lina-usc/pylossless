"""Module file contains Config class which defines procedures."""

import sys
from pathlib import Path
import yaml


# TODO: Add parameters and references for below functions.
# TODO: Add magic methods and constructors for this class.
class Config(dict):
    """Representation of configuration files for a pipeline procedure."""

    DEFAULT_CONFIG_PATH = (Path(__file__).parent.parent /
                           "assets" / "ll_default_config.yaml")

    def read(self, file_name):
        """Read a saved pylossless config file."""
        file_name = Path(file_name)
        if not file_name.exists():
            raise FileExistsError(f'Configuration file {file_name.absolute()} '
                                  'does not exist')

        with file_name.open("r") as init_variables_file:
            self.update(yaml.safe_load(init_variables_file))

        return self

    def load_default(self):
        """Get the default pylossless config file."""
        self.read(Config.DEFAULT_CONFIG_PATH)
        return self

    def save(self, file_name):
        """Save the current Config object to disk."""
        file_name = Path(file_name)
        with file_name.open("w") as init_variables_file:
            yaml.dump(dict(self), init_variables_file,
                      indent=4, sort_keys=True)

    def print(self):
        """Print the Config contents."""
        yaml.dump(dict(self), sys.stdout, indent=4, sort_keys=True)
