import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyLossless"
copyright = "2023, Huberty, Scott; O'Reilly, Christian; Desjardins, James"
author = "Huberty, Scott; O'reilly, Christian"
release = "0.1"


# Point Sphinx.ext.autodoc to the our python modules (two parent directories
# from this dir)
sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Autodoc for: import modules from pipeline, extract docstring
# numpydoc for: building doc from Numpy formatted docstrings
# ext.todo: so we can use the ..todo:: directive
# gallery for: building tutorial .rst files from python files
# sphinxemoji So we can use emoji's in docs.
# sphinx design to support certain directives, like ::grid etc.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinxemoji.sphinxemoji",
    "sphinx_design",
]

# Allows us to use the ..todo:: directive
todo_include_todos = True

# sphinx_gallery extension Settings
# Source directory of python file tutorials and the target
# directory for the converted rST files
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to save tutorials
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "mne": ("https://mne.tools/dev", None),
    "mne_icalabel": ("https://mne.tools/mne-icalabel/dev", None),
    "mne_bids": ("https://mne.tools/mne-bids/dev", None),
    "pylossless": ("https://pylossless.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
# TODO: add a svg file for the logo
# html_theme_options = {
#    "light_logo": "logo-lightmode_color.png",
#    "dark_logo": "logo_white.png",
# }

# user made CSS to customize look
html_css_files = [
    "css/custom.css",
]

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {"index": ["search-field.html", "sidebar-nav-bs", "globaltoc.html"]}

# NumPyDoc configuration -----------------------------------------------------

numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_validate = True
# Only generate documentation for public members
autodoc_default_flags = ["members", "undoc-members", "inherited-members"]

numpydoc_class_members_toctree = False

numpydoc_xref_aliases = {
    # Python
    "file-like": ":term:`file-like <python:file object>`",
    "iterator": ":term:`iterator <python:iterator>`",
    "path-like": ":term:`path-like`",
    "array-like": ":term:`array_like <numpy:array_like>`",
    "Path": ":class:`python:pathlib.Path`",
    "bool": ":class:`python:bool`",
    "dictionary": ":class:`python:dict`",
}
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    "LosslessPipeline",
}
