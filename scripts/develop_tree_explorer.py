"""
Script to run a Dash application for exploring the taxonomy tree visualization.

This script initializes a Dash app that displays an interactive phylogenetic tree.
It handles user interactions such as clicking on nodes to expand/collapse branches
and resetting the view. It relies on the `TreeVizUtils` class for tree
processing and visualization logic.
"""

from pathlib import Path
from typing import cast

from dash import Dash, Input, Output, State, ctx, dcc, html

from vaast_app.utils.tree_viz_utils import TreeVizUtils


def _main() -> None:
    # Initialize Utils
    nwk_path = Path(__file__).parent / "full_collection_tree.nwk"
    if not nwk_path.exists():
        print(f"Error: {nwk_path} not found. Run generate_full_collection_tree.py first.")
        return

    tree_viz_utils = TreeVizUtils(nwk_path)

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("VAAST Taxonomy Explorer"),
            html.Button("Reset View", id="reset-btn", n_clicks=0),
            html.Div(id="breadcrumbs", style={"padding": "10px"}),
            dcc.Graph(id="tree-graph", style={"height": "90vh"}),
            # Store current root info
            dcc.Store(id="current-root", data={"name": "Bacteria", "rank": "superkingdom"}),
        ]
    )

    @app.callback(
        Output("tree-graph", "figure"),
        Output("current-root", "data"),
        Output("breadcrumbs", "children"),
        Input("tree-graph", "clickData"),
        Input("reset-btn", "n_clicks"),
        State("current-root", "data"),
    )
    def update_tree(click_data, reset_clicks, current_root_data):
        triggered_id = ctx.triggered_id or "initial"

        return tree_viz_utils.process_interaction(cast(str, triggered_id), click_data, current_root_data)

    app.run(debug=True)


if __name__ == "__main__":
    _main()
