# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "audiozen"
copyright = "2023, HAO, Xiang"
author = "HAO, Xiang"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../../audiozen"]
autodoc_typehints = "description"

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_attr_annotations = False
napoleon_use_param = True
napoleon_type_aliases = {
    "NDArray": ":term:`array-like <array_like>`",
    "array_like": ":term:`array_like`",
}


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,  # edit on Github, see https://github.com/readthedocs/sphinx_rtd_theme/issues/529
    "github_user": "haoxiangsnr",
    "github_repo": "audiozen",
    "github_version": "main",
}
html_static_path = ["_static"]
