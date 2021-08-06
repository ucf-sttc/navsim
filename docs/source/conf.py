# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

with open('../../navsim-lab/navsim/version.txt', 'r') as vf:
    version = vf.read().strip()
project = f'NavSim'
copyright = '2021, STTC, UCF'
author = 'STTC, UCF'
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.autosectionlabel',
              'sphinx_rtd_theme',
              #'recommonmark',
              'myst_parser',
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # 'alabaster'
html_theme_options = {
#    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
#    'analytics_anonymize_ip': False,
#    'logo_only': False,
#    'display_version': True,
#    'prev_next_buttons_location': 'bottom',
#    'style_external_links': False,
#    'vcs_pageview_mode': '',
#    'style_nav_header_background': 'white',
    # Toc options
#    'collapse_navigation': True,
#    'sticky_navigation': True,
    'navigation_depth': 2,
#    'includehidden': True,
#    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#import recommonmark
#from recommonmark.transform import AutoStructify


# At the bottom of conf.py
def setup(app):
    app.add_config_value('recommonmark_config', {
        # 'url_resolver': lambda url: github_doc_root + url,
        # 'auto_toc_tree_section': 'Contents',
    }, True)

#    app.add_transform(AutoStructify)


# -- Options for Latex output -------------------------------------------------
latex_elements = {
    'releasename': f'{version}',
    'fncychap': r'\usepackage[Sonny]{fncychap}'
}
latex_theme = 'manual'
# latex_toplevel_sectioning = 'part'
