.. |min_python_version| replace:: 3.7


Installation
============

To stay up to date with the latest version of pyLossless, we recommend that you install
the package from the github repository. This will allow you to easily update to the
latest version of pyLossless as we continue to develop it.

.. hint::
    To use pyLossless you need to have the ``git`` command line tool installed.
    If you are not sure, see this
    `tutorial
    <https://mne.tools/stable/install/contributing.html>`__


Once you have git installed and configured, and before creating your local copy
of the codebase, go to the `PyLossless GitHub <https://github.com/lina-usc/pylossless>`_
page and create a
`fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ into your GitHub
user account.

****************************************
Install via :code:`pip` or :code:`conda`
****************************************

Pylossless requires Python version |min_python_version| or higher. If you
need to install Python, please see `MNE-Pythons guide to installing Python
<https://mne.tools/stable/install/manual_install_python.html#install-python>`__

Create a virtual environment and install Pylossless into it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We suggest to install Python into its own ``conda`` environment.

Run in your terminal:

.. code-block:: console

    $ conda create --name=pylossless
    $ conda activate pylossless

This will create a new ``conda`` environment named ``pylossless`` (you can
adjust this by passing a different name via ``--name``) and install all
dependencies into it.

The second command ( ``conda activate`` ) will activate the environment so
that any python packages are installed and isolated in this environment.

Clone the pyLossless github repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package is not yet deployed on PyPI. It can be installed directly from
the github repository.

First, go to a directory in your terminal that you would like to copy the
pyLossless git repository into (for example Documents, or a new folder named
github_repos). Then run:

.. code-block:: console

   $ git clone git@github.com:lina-usc/pylossless.git 

Install Pylossless and it's dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: console

   $ pip install --editable ./pylossless

or via :code:`conda`:

.. code-block:: console

   $ conda develop ./pylossless


That's it! You are now ready to use pyLossless.

Additional Requirements for Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan on contributing to the development of pyLossless, you will need to install
some additional dependencies so that you can run tests and build the documentation
locally. The code below will install the additional dependencies as well as the
pre-commit hooks that we use to ensure that all code is formatted correctly. Make sure
that you have activated your ``pylossless`` environment and are inside the pylossless
git repository directory, before running the code below:

.. code-block:: console

   $ pip install -r requirements_testing.txt
   $ pip install -r docs/requirements_doc.txt
   $ pre-commit run -a

PyLossless uses `black <https://github.com/psf/black>`_ style formatting. If you are
using Visual Studio Code, you can also install the black extension to automatically
format your code. See the instructions at this
`link 
<https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0>`_
