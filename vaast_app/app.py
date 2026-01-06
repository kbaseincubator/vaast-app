"""
Genetic Tools ChatBot application
"""

import logging
import os

import dash_bootstrap_components as dbc
import diskcache
from dash import Dash, DiskcacheManager, dcc, html, page_container

from vaast_app.utils.ncbi_taxa import get_ncbi_taxa

logging.basicConfig(level=logging.INFO)

get_ncbi_taxa()
cache = diskcache.Cache("./.cache")
background_callback_manager = DiskcacheManager(cache)


app = Dash(
    __name__,
    title="VAAST by KBase",
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    background_callback_manager=background_callback_manager,
    use_pages=True,
)
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id="store-main-query", data={"query": []}),
        page_container,
    ],
    className="g-0",
)


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=os.environ.get("PORT", "8000"))
    except KeyboardInterrupt:
        ...
    except BaseException as err:
        raise err
