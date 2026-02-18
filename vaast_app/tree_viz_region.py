"""Module for the Tree Visualization Region."""

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html
from dash.development.base_component import Component

from vaast_app.render import Render
from vaast_app.utils.tree_viz_utils import TreeVizUtils

RANK_HIERARCHY = ["class", "order", "family", "genus", "species"]


class TreeVizRegion(Render):
    """
    Region for visualizing the full taxonomy tree with zoom features.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the TreeVizRegion."""
        super().__init__(*args, **kwargs)
        self.tree_utils = TreeVizUtils(Path("scripts/full_collection_tree.nwk"))

        self.app.callback(
            Output("tree-viz-graph", "figure"),
            Output("tree-viz-store", "data"),
            Output("tree-viz-breadcrumbs", "children"),
            Input("tree-viz-graph", "clickData"),
            Input("tree-viz-reset-btn", "n_clicks"),
            Input("tree-viz-back-btn", "n_clicks"),
            State("tree-viz-store", "data"),
            prevent_initial_call=True,
        )(self._update_tree_view)

    def _set_layout(self) -> Component:
        initial_data = {"name": "Bacteria", "rank": "superkingdom", "history": []}

        leaf_rank = "order"
        color_rank = "phylum"
        tree = self.tree_utils.get_collapsed_tree(leaf_rank, root_tax_name="Bacteria")
        initial_fig = self.tree_utils.generate_figure(tree, color_by_rank=color_rank, label_rank=color_rank)

        return dbc.Container(
            [
                html.H4("Taxonomy Explorer"),
                dcc.Store(id="tree-viz-store", data=initial_data),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Back",
                                    id="tree-viz-back-btn",
                                    size="sm",
                                    color="secondary",
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Reset View",
                                    id="tree-viz-reset-btn",
                                    size="sm",
                                    color="secondary",
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span("Click on a node to zoom in."),
                            className="align-self-center ms-2 text-muted",
                            style={"fontSize": "0.9em"},
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    id="tree-viz-breadcrumbs",
                    style={"padding": "10px"},
                    children="Current Root: Bacteria (superkingdom)",
                ),
                dcc.Loading(
                    id="loading-tree-graph",
                    type="default",
                    children=dcc.Graph(
                        id="tree-viz-graph", figure=initial_fig, config={"displayModeBar": True, "scrollZoom": True}
                    ),
                ),
            ],
            className="border rounded p-3 mt-3 mb-3",
        )

    def _update_tree_view(self, click_data, reset_clicks, back_clicks, current_store):
        triggered_id = ctx.triggered_id

        mock_id = "initial"
        if triggered_id == "tree-viz-reset-btn":
            mock_id = "reset-btn"
        elif triggered_id == "tree-viz-back-btn":
            mock_id = "back-btn"
        elif triggered_id == "tree-viz-graph":
            mock_id = "tree-graph"

        return self.tree_utils.process_interaction(mock_id, click_data, current_store)
