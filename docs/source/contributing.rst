Contributing
==============

We are glad you are here! Contributions to this package are always welcome.
Read on to learn more about the contribution process and package design.

Building and Contributing to the Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyLossless uses `Sphinx <https://www.sphinx-doc.org/en/master/>`__ for building
documentation. Specifically, we use the `PyData-sphinx-theme 
<https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html>`__ as a
design template for our page. We also use the `Napolean 
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html>`__
and
`autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`__
sphinx extensions to build API documentation directly from the pyLossless
docstrings. Finally, we use the `sphinx-gallery 
<https://sphinx-gallery.github.io/stable/index.html>`__ extension to build
documentation tutorials directly from python files.

Before building the docs, make sure you install additional requirements in the
``requirements_doc.txt`` file at the root of the ``pylossless`` directory.

.. code-block:: console

    $ pip install requirements_doc.txt

To help understand the layout of the source documents, here is the layout of
the pylossless package:

.. code-block:: python

    """
    pylossless
    ├── docs
    |   ├── make.bat
    │   └── Makefile
    |   └──build
    |   └──source
    |   |   ├── _images
    |   |   ├── _static
    |   |   └── API
    |   |       ├── API_index.rst
    |   |       ├── bids_funcs.rst
    |   |       ├── config_class.rst
    |   |       ├── flag_classes.rst
    |   |       ├── lossless_class.rst
    |   |       ├── pipeline_funcs.rst
    |   |   └── generated
    |   |       ├── index.rst
    │   |   ├── conf.py
    │   |   ├── index.rst
    |   |   ├── install.rst
    |   |   ├── implementation.rst
    |   |   ├── contributing.rst
    |   └── examples
    |        ├── usage.py
    |        └── README.txt
    |   
    ├── pylossless
    │   ├── __init__.py
    │   └── pipeline.py
    """

For building documentation, all you will need is in the ``docs`` folder. Most
of what you will need is directly in the ``docs/source`` directive. The
``Makefile`` and ``makefile.bat`` provide a useful ``make`` command line
tool to use to build the documentation **from the** ``pylossless/docs``
**directory**

The homepage is documented in ``docs/source/index.rst``. It links to the installation
instructions, the implementation page, and this page (contributing), using
the ``.. toctree::`` sphinx directive:

- installation instructions are in ``docs/source/install.rst``.
- Implementation documentation is in ``docs/source/implementation.rst``.
- Contributing documentation is in ``docs/source/contributing.rst``.

If you create a new rST file and want to link to it from the Homepage,
you should add the rST file to ``docs/source``, and then add the name of the
rST file (without the extension) to the ``toctree`` directive in
``docs/source/index.rst``.

pyLossless tutorials can be edited and created in ``docs/examples``. The
``sphinx-gallery`` sphinx extension will take any python files in this folder,
and generate rST files from them, outputting them into
``docs/source/generated`` in a new folder named ``autotutorials``. 
You do not need to interact with this generated ``autotutorials folder``.

If you create a new tutorial, you just need to place the .py file in
``docs/examples``For example, ``docs/examples/my_lossless_tutorial.py``.


Building the docs locally
^^^^^^^^^^^^^^^^^^^^^^^^^

Once you are ready to build the docs, you can use the ``make`` command line
tool. make sure you are in the ``pylossless/docs`` directory.

.. code-block:: console

    $ make clean
    $ make html

``make clean`` clears out any generated documentation in ``docs/build``, and
it is generally good practice to clear this between runs of ``make html`` If
you are working on documentation, to avoid errors when building the docs.

``make html`` will build the documentation.

Viewing the docs locally
^^^^^^^^^^^^^^^^^^^^^^^^

The built documentation is placed in ``docs/build``. You should not
change any files in this directory. If you want to view the documentation
locally, simply click on the ``docs/build/html/index.html`` file from your
file browser or open it with the command line:

If you are in the ``docs`` directory:

.. code-block:: console

    $ open build/html/index.rst

Settings for documentation template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Important settings for this template are located in ``docs/source/conf.py``.
You will not need to (and should not) change these settings in most situations.

To build the docstrings, the following lines in the ``conf.py`` file are used
to point ``sphinx-autodoc`` to the docstrings in the pylossless modules

.. code-block:: python
    :caption: conf.py
    
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../..'))

The following code is used to set our theme and point sphinx to our logo file:

.. code-block:: python
    :caption: conf.py
    
    html_theme = 'pydata_sphinx_theme'
    html_static_path = ['_static']
    html_theme_options = {
    "logo": {
        "image_light": "logo-light_mode.png",
        "image_dark": "logo-dark_mode.png",
    }
    }