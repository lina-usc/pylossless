.. |min_python_version| replace:: 3.7


Installation
============

****************************************
Install via :code:`pip` or :code:`conda`
****************************************

.. hint::
    To use pyLossless you need to have the ``git`` command line tool installed.
    If you are not sure, see this
    `tutorial
    <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__

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