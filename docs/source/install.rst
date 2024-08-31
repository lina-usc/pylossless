.. |min_python_version| replace:: 3.8


Installation
============
Pylossless requires Python version |min_python_version| or higher.
To stay up to date with the latest version of pyLossless, we recommend that you install
the package from the github repository. This will allow you to easily update to the
latest version of pyLossless as we continue to develop it.

.. hint::
    To use pyLossless you need to have the ``git`` command line tool installed.
    If you are not sure, see this
    `tutorial
    <https://mne.tools/stable/install/contributing.html>`__


***********************
Install via :code:`pip`
***********************

Clone the pyLossless github repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package can be installed directly from the github repository.

First, go to a directory in your terminal that you would like to copy the
pyLossless git repository into (for example Documents, or a new folder named
github_repos). Then run:

.. code-block:: console

   $ git clone https:@github.com:[YOUR-GITHUB-USERNAME]/pylossless.git 

Of course, replace ``[YOUR-GITHUB-USERNAME]`` with your actual GitHub username.
For example, for me this would be ``https://github.com/scott-huberty/pylossless.git``.

.. Note::
   Make sure you have created a fork of the
   `pyLossless repository <https://github.com/lina-usc/pylossless>`_ in your GitHub
   account before running the code above.


Install Pylossless and it's dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: console

   $ pip install --editable ./pylossless


.. Note::
   The ``--editable`` flag is optional, but it allows you to easily update to the
   latest version of pyLossless as we continue to develop it. We also recommend
   installing pyLossless in a virtual environment.


That's it! You are now ready to use pyLossless.

Install Extra dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

The PyLossless pipeline uses the
`MNE-ICALabel <https://mne.tools/mne-icalabel/stable/index.html>`_
package, which uses deep learning to automatically label independent components.
MNE-ICALabel requires that you have either `PyTorch <https://pytorch.org/>`_ or
OnnxRuntime installed, but does not install them for you. We recommend simply
installing PyTorch, as it is the more popular of the two packages. 

.. code-block:: console

   $ pip install torch


.. Note::
   As of this writing, PyTorch is not available on Python 3.12. Please use an earlier
   version of Python, such as Python 3.9

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

PyLossless uses `ruff <https://docs.astral.sh/ruff/>`_ style formatting. If you are
using Visual Studio Code, you can also install the ruff extension to automatically
format your code. See the instructions at this
`link <https://docs.astral.sh/ruff/editors/setup/>`_
