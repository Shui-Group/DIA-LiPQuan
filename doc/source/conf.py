# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Add path to your package

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lipana"
copyright = "2025, shui lab"
author = "shui lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core extension for API docs
    "sphinx.ext.viewcode",  # Add links to view source code
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx_autodoc_typehints",  # Support for type hints
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"  # Include both class and __init__ docstrings
