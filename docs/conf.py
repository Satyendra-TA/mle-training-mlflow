# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, \
# see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "House-Price-Predictor"
copyright = "2024, Satyendra"
author = "Satyendra"
release = "v0.2"

# -- General configuration -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for auto docstring extraction
    "sphinx.ext.napoleon",  # For supporting NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.mathjax",  # Render math via JS
    "numpydoc",  # Handles NumPy-style docstrings
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Use the Read the Docs theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
