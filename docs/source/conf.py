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

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

with open('../../version.txt', 'r') as vf:
    version = vf.read().strip()
project = 'navsim'
copyright = '2021, STTC, UCF'
author = 'STTC, UCF'
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosectionlabel',
              'sphinx_copybutton',
              'sphinx_togglebutton',
              'sphinx_book_theme',
              # 'sphinxcontrib.katex',
              # 'sphinx_copybutton',
              # 'sphinx_panels',
              'myst_parser',
              'sphinxcontrib.programoutput',
              ]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_default_flags = []
autodoc_default_options = {
    'member-order': 'alphabetical',
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': False,
    'inherit-docstrings': False,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'  # 'alabaster'
html_theme_options = {
    "home_page_in_toc": True,
    "show_navbar_depth": 3,  # = pydata show_nav_level
    "navigation_depth": 3,   # pydata maxdepth
    "show_nav_level": 3,     # pydata
    "show_toc_level": 3,
    "secondary_sidebar_items": [],
    #    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google
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
    #'navigation_depth': 2,
    #'includehidden': True,
    #    'titles_only': False
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_show_sourcelink = False
html_copy_source = False
html_title = f'{project} v{version} docs'


# At the bottom of conf.py
def setup(app):
    pass
    # app.add_config_value('recommonmark_config', {
    # 'url_resolver': lambda url: github_doc_root + url,
    # 'auto_toc_tree_section': 'Contents',
    # }, True)


#    app.add_transform(AutoStructify)


# -- Options for Latex output -------------------------------------------------
latex_elements = {
    'releasename': f'{version}',
    'fncychap': r'\usepackage[Sonny]{fncychap}'
}
latex_theme = 'manual'
# latex_toplevel_sectioning = 'part'
myst_heading_anchors = 3

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
