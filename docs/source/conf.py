# -- Project information -----------------------------------------------------
project = "audiozen"
author = "HAO Xiang <haoxiangsnr@gmail.com>"
copyright = "2023, HAO Xiang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# -- Extension configuration -------------------------------------------------
myst_enable_extensions = ["colon_fence"]
myst_number_code_blocks = ["python"]
autodoc_default_options = {"member-order": "bysource"}
autodoc_mock_imports = ["torch"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,  # edit on Github, see https://github.com/readthedocs/sphinx_rtd_theme/issues/529
    "github_user": "haoxiangsnr",
    "github_repo": "audiozen",
    "github_version": "main/docs/source/",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
