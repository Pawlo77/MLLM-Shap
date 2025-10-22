# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

cur_path: str = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, cur_path)
sys.path.insert(0, os.path.join(cur_path, "..", "src"))

project = 'mllm-shap'
copyright = '2025, Paweł Pozorski, Jakub Muszyński'
author = 'Paweł Pozorski, Jakub Muszyński'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "myst_parser",
    "extensions.custom_skip",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Prefix each section label with the document name to avoid duplicates
autosectionlabel_prefix_document = True
autodoc_inherit_docstrings = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__,__call__",
    "inherited-members": True,
    "show-inheritance": True,
    "exclude-members": "_deprecated",
    "private-members": True,
}

# suppress all docutils-related warnings, as we use slightly non-standard syntax in some places
suppress_warnings = [
    "docutils",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    'custom.css',
]
