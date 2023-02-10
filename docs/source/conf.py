# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyLossless'
copyright = "2023, Huberty, Scott; O'Reilly, Christian; Desjardins, James"
author = "Huberty, Scott; O'reilly, Christian"
release = '0.1'


# Point Sphinx.ext.autodoc to the our python modules (two parent directories
# from this dir)
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Autodoc for: import modules from pipeline, extract docstring
# napoleon for: building doc from Numpy formatted docstrings 
# ext.todo: so we can use the ..todo:: directive
# gallery for: building tutorial .rst files from python files
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx_gallery.gen_gallery'] 

# Allows us to use the ..todo:: directive
todo_include_todos = True

# sphinx_gallery extension Settings
# Source directory of python file tutorials and the target
# directory for the converted rST files 
sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': './generated/auto_tutorials',  # path to where to save gallery generated output
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme' #'alabaster'
html_static_path = ['_static']
html_theme_options = {
   "logo": {
      "image_light": "logo-light_mode.png",
      "image_dark": "logo-dark_mode.png",
   }
}

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html", "sidebar-nav-bs", 'globaltoc.html']
}
