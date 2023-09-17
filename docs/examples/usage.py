"""

Run pyLossless using an example dataset.
========================================

This notebook demonstrate how the PyLossless Quality Control interface can be
integrated in a Jupyter Notebook. This can even be done on a virtualized
platform, like Google Colab! To try it, you can run this notebook from
`here!
<https://colab.research.google.com/github/lina-usc/pylossless/blob/main/
notebooks/qc_example.ipynb>`__.
"""

# %%
# This tutorial assumes you have pylossless installed.
# ----------------------------------------------------
# Then, we import the function we need.
# For use in jupyter notebooks, We just need to import a single function.
from pylossless.dash.app import get_app

app = get_app(kind="jupyter")
app.run_server(mode="inline")
