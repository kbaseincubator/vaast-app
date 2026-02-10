from pathlib import Path
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
from dash.development.base_component import Component
from vaast_app.render import Render
from vaast_app.utils.tree_viz_utils import TreeVizUtils

RANK_HIERARCHY = ["class", "order", "family", "genus", "species"]


class TreeVizRegion(Render):
    """
    Region for visualizing the full taxonomy tree with zoom features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize utils - assume file exists at scripts/full_collection_tree.nwk relative to app run
        # Ideally this path should be config/env
        self.tree_utils = TreeVizUtils(Path("scripts/full_collection_tree.nwk"))

        self.app.callback(
            Output("tree-viz-graph", "figure"),
            Output("tree-viz-store", "data"),
            Input("tree-viz-graph", "clickData"),
            Input("tree-viz-reset-btn", "n_clicks"),
            State("tree-viz-store", "data"),
            prevent_initial_call=True,
        )(self._update_tree_view)

    def _set_layout(self) -> Component:
        # Initial View: Collapsed at Class level
        initial_rank = "class"
        initial_tree = self.tree_utils.get_collapsed_tree(target_rank=initial_rank)
        initial_fig = self.tree_utils.generate_figure(initial_tree)
        initial_fig.update_layout(height=600, title="Taxonomy Tree (Zoom: Class)")

        return dbc.Container(
            [
                html.H4("Taxonomy Explorer"),
                dcc.Store(id="tree-viz-store", data={"rank": initial_rank, "root": None}),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Reset Zoom", id="tree-viz-reset-btn", size="sm", color="secondary"),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span("Click on a node to zoom in. Red nodes indicate collapsed groups."),
                            className="align-self-center ms-2 text-muted",
                            style={"fontSize": "0.9em"},
                        ),
                    ],
                    className="mb-2",
                ),
                dcc.Graph(id="tree-viz-graph", figure=initial_fig, config={"displayModeBar": True, "scrollZoom": True}),
            ],
            className="border rounded p-3 mt-3 mb-3",
        )

    def _update_tree_view(self, click_data, reset_clicks, current_store):
        triggered_id = ctx.triggered_id

        if triggered_id == "tree-viz-reset-btn":
            rank = "class"
            root = None
            tree = self.tree_utils.get_collapsed_tree(target_rank=rank)
            fig = self.tree_utils.generate_figure(tree)
            fig.update_layout(height=600, title=f"Taxonomy Tree (Zoom: {rank.capitalize()})")
            return fig, {"rank": rank, "root": root}

        if triggered_id == "tree-viz-graph" and click_data:
            # Logic to zoom in
            point = click_data["points"][0]
            customdata = point.get("customdata")
            if not customdata:
                return no_update, no_update

            # customdata = [name, rank, count]
            name_line, rank_line, _ = customdata

            current_rank = current_store.get("rank", "class")

            # Determine next rank
            try:
                current_rank_idx = RANK_HIERARCHY.index(rank_line.lower())
                next_rank_idx = current_rank_idx + 1
                if next_rank_idx >= len(RANK_HIERARCHY):
                    return no_update, no_update  # Max depth
                next_rank = RANK_HIERARCHY[next_rank_idx]
            except ValueError:
                # If rank not in hierarchy, maybe do nothing
                return no_update, no_update

            # New State
            new_root = name_line

            tree = self.tree_utils.get_collapsed_tree(target_rank=next_rank, root_tax_name=new_root)
            fig = self.tree_utils.generate_figure(tree)
            fig.update_layout(height=600, title=f"Taxonomy Tree (Root: {new_root}, Zoom: {next_rank.capitalize()})")

            return fig, {"rank": next_rank, "root": new_root}

        return no_update, no_update
