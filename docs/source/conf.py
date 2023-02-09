# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyLossless'
copyright = "2023, Huberty, Scott; O'reilly, Christian"
author = "Huberty, Scott; O'reilly, Christian"
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo'] #['sphinx_gallery.gen_gallery',]
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme' #'alabaster'
html_static_path = [] # ['_static']

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html", "sidebar-nav-bs", 'globaltoc.html']
}
