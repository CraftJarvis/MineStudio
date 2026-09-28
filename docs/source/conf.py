'''
Date: 2024-11-28 17:46:44
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-29 14:09:34
FilePath: /MineStudio/docs/source/conf.py
'''
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sphinx.application import Sphinx
from sphinx.locale import _
import pydata_sphinx_theme
sys.path.append(str(Path(".").resolve()))
# sys.path.insert(0, str(Path('..', '../../minestudio').resolve()))

from custom_directives import generate_versions_json

sys.path.insert(0, os.path.abspath('../../src'))

project = 'MineStudio'
copyright = str(datetime.now().year) + ", The CraftJarvis Team"
author = 'The CraftJarvis Team'
release = '1.1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    # "autoapi.extension",
    # For extension examples and demos
    "myst_parser",
    "ablog",
    "jupyter_sphinx",
    "sphinxcontrib.youtube",
    "nbsphinx",
    # "numpydoc",
    "sphinx_togglebutton",
    # "jupyterlite_sphinx",
    "sphinx_favicon",
    "sphinx_multiversion",
    "_extension.gallery_directive",
    "_extension.component_directive",
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

sys.path.insert(0, os.path.abspath('../../src/minestudio'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# htmp_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    "custom.css"
]

html_theme_options = {
  "header_links_before_dropdown": 7,
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CraftJarvis/MineStudio",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/minestudio",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "ArXiv",
            "url": "https://arxiv.org/pdf/2411.18293",
            "icon": "fa-custom fa-arxiv",
        }
  ], 
  "navbar_align": "left",
  "show_toc_level": 1,
  "navbar_center": ["version-switcher", "navbar-nav"],  
  "logo": {
    "text": "MineStudio",
    "image_light": "_static/logo-no-text-light.svg", 
    "image_dark": "_static/logo-no-text-light.svg",
  },
  "navbar_start": ["navbar-logo"],  # 在导航栏显示 Logo
  "switcher": {
        "json_url": "https://craftjarvis.github.io/MineStudio/_static/version.json",
        "version_match": "latest",
   },
   "check_switcher": True,
}

html_title = "MineStudio"
html_favicon = "_static/logo-no-text-light.svg"

html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]

# -- auto api ----------------------------------------------------------------
autoapi_add_toctree_entry = False
autoapi_dirs = ['../../src/minestudio']
autoapi_ignore = ['*minerl*', '*tests*', '*tutorials*', '*utils*']

autodoc_mock_imports = ['torch', 'absl', 'cv2', 'huggingface_hub', 'lmdb', 'gymnasium', 'lightning', 'omegaconf', 'tree', 'einops', 'gym', 'gym3', 'coloredlogs', 'daemoniker', 'av', 'minecraft_data', 'torchvision', 'timm', 'transformers', 'x_transformers', 'tqdm']
autodoc_inherit_docstring = False
autosummary_generate = False
numpydoc_show_class_members = False 
# -- application setup ----------
# ---------------------------------------------


def setup_to_main(
    app: Sphinx, pagename: str, templatename: str, context, doctree
) -> None:
    """
    Add a function that jinja can access for returning an "edit this page" link
    pointing to `main`.
    """

    def to_main(link: str) -> str:
        """
        Transform "edit on github" links and make sure they always point to the
        main branch.

        Args:
            link: the link to the github edit interface

        Returns:
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("edit")
        return "/".join(links[: idx + 1]) + "/main/" + "/".join(links[idx + 2 :])

    context["to_main"] = to_main


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parallel parameters set to ``True``.
    """

    generate_versions_json()

    app.connect("html-page-context", setup_to_main)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

# setup multiversion
smv_tag_whitelist = r'^.*$' 
smv_released_pattern = r'^refs/tags/.*$'