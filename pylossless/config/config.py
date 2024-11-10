# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

import sys
from pathlib import Path
import yaml


# TODO: Add parameters and references for below functions.
# TODO: Add magic methods and constructors for this class.
class ConfigMixin(dict):
    """Base configuration file class for pipeline procedures."""

    DEFAULT_CONFIG_PATH = (
        Path(__file__).parent.parent / "assets"
    )

    def read(self, file_name):
        """Read a saved pylossless config YAML file."""
        file_name = Path(file_name)
        if not file_name.exists():
            raise FileExistsError(
                f"Configuration file {file_name.absolute()} " "does not exist"
            )

        with file_name.open("r") as init_variables_file:
            self.update(yaml.safe_load(init_variables_file))

        return self

    def save(self, file_name):
        """Save the current config object to disk as a YAML file.

        Parameters
        ----------
        file_name : str | pathlib.Path
            The file name to save the :class:`~pylossless.Config` object to.
        """
        file_name = Path(file_name)
        with file_name.open("w") as init_variables_file:
            yaml.dump(dict(self), init_variables_file, indent=4, sort_keys=True)

    def print(self):
        """Print the Config contents."""
        yaml.dump(dict(self), sys.stdout, indent=4, sort_keys=True)


class Config(ConfigMixin):
    """Representation of configuration file for running the pipeline."""

    def load_default(self, kind="adults"):
        """Get the default pylossless config file.

        Parameters
        ----------
        kind : str | pathlib.Path
            Can be either 'adults' or 'infants'. Default to 'adults'.
        """
        path = Config.DEFAULT_CONFIG_PATH / f"ll_default_config_{kind}.yaml"
        if not path.exists():
            raise ValueError(f"No default configuration for kind '{kind}'.")
        self.read(path)
        return self
