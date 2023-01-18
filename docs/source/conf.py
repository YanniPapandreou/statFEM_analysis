# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'statFEM_analysis'
copyright = '2022, Yanni Papandreou'
author = 'Yanni Papandreou'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../../statFEM_analysis'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosummary',
    'nbsphinx',
]

# autodoc_mock_imports = ["dolfin", "numpy", "scipy", "joblib", "multiprocessing"]
autoclass_content = 'both'
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'
templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints', 'version.py']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
html_static_path = []
