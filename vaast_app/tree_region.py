"""Tree region"""

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component
from ete3 import NCBITaxa
from phylotree_dash import PhylotreeDash

from vaast_app.render import Render
from vaast_app.utils.bacdive_utils import BacdiveMetadata
from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload
from vaast_app.utils.genetic_tool_wrapper import GeneticToolDB
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.tree_utils import TreeJSON, TreeResultData, TreeUtils
from vaast_app.utils.type_utils import TaxID


class TreeRegion(Render):
    """Tree region"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app.callback(
            Output("tree-loading", "children", allow_duplicate=True),
            Output("results-tree", "newick"),
            Output("results-tree", "metadata"),
            Output("results-tree", "selected_nodes"),
            Input("validated-results", "data"),
            Input("add-selection-button", "n_clicks"),
            Input("prune-selection-button", "n_clicks"),
            Input("add-strain-selection-button", "n_clicks"),
            State("results-tree", "selected_nodes"),
            State("chatbot-provided", "data"),
            prevent_initial_call=True,
        )(self._visualize_selection)

        self.app.callback(
            Output("tree-loading", "children", allow_duplicate=True),
            Output("results-tree", "max_radius"),
            Input("radius-slider", "value"),
            prevent_initial_call=True,
        )(self._update_radius)

    def _set_layout(self) -> Component:
        # TODO: Dynamically set rings based on tool types
        # TODO: Initial query
        return dbc.Container(
            [
                dbc.Container(
                    [
                        PhylotreeDash(
                            id="results-tree",
                            newick="",
                            metadata={},
                            ranks=self.bacdive_searcher.ranks,
                            colors=["#F78E1E", "#FFD200", "#5E9732", "#C1CD23", "#007DC3", "#72CCD2"],
                            starting_rank="species",
                            trait_bubble_size=3,
                            left_offset=25,
                            selected_nodes=[],
                            max_radius=400,
                            display_visualize_button=False,
                            display_rings=[True, True, True, True],
                            rings=[
                                {
                                    "kind": "categorical",
                                    "colors": ["white", color],
                                    "legend_label": tool.database_name,
                                    "legend_color": color,
                                }
                                for color, tool in zip(
                                    ["blue", "green", "orange", "red"],
                                    self.genetic_tools + [GeneticToolDB("VAAST", "", "", "", "", Path())],
                                    strict=True,
                                )
                            ],
                            height="71vh",
                            width="100%",
                        )
                    ],
                    className="g-0",
                    id="results-tree-container",
                ),
                dbc.Container(
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    id="add-strain-selection-button",
                                    children="View selected strains",
                                ),
                                className="g-0 d-flex justify-content-center",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    id="add-selection-button",
                                    children="Add at selection",
                                ),
                                className="g-0 d-flex justify-content-center",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    id="prune-selection-button",
                                    children="Prune selection",
                                ),
                                className="g-0 d-flex justify-content-center",
                            ),
                            dbc.Col(
                                [
                                    html.Div("Tree radius", className="h5 ps-2"),
                                    html.Div(
                                        dcc.Slider(
                                            100,
                                            1000,
                                            100,
                                            value=400,
                                            id="radius-slider",
                                            marks=None,
                                        )
                                    ),
                                ],
                                className="g-0 justify-content-center",
                            ),
                        ],
                        className="d-flex justify-content-between g-0",
                    ),
                    className="mt-2",
                ),
            ],
            className="g-0 h-100",
        )

    @staticmethod
    def _pad_at(
        updated_nodes: set[TaxID], taxid: int, ncbi_client: NCBITaxa, bd_ids: set[TaxID], pad_size: int
    ) -> None:
        taxid_lineage = ncbi_client.get_lineage(taxid)
        # Parent taxid of current
        parent_taxid = taxid_lineage[taxid_lineage.index(taxid) - 1]
        n_change = pad_size
        if pad_size > 0:
            for desc_tax_id in ncbi_client.get_descendant_taxa(
                parent_taxid, rank_limit="species", collapse_subspecies=True
            ):
                if n_change == 0:
                    break
                if desc_tax_id not in bd_ids and desc_tax_id not in updated_nodes:
                    updated_nodes.add(desc_tax_id)
                    n_change -= 1
        else:
            for desc_tax_id in ncbi_client.get_descendant_taxa(parent_taxid):
                if n_change == 0:
                    break
                if desc_tax_id not in bd_ids and desc_tax_id in updated_nodes:
                    updated_nodes.remove(desc_tax_id)
                    n_change += 1

    @staticmethod
    def _collect_subspecies(updated_nodes: set[TaxID], taxid: int, ncbi_client: NCBITaxa, pad_size: int) -> None:
        assert pad_size > 0
        updated_nodes.add(TaxID(taxid))
        n_change = pad_size
        taxid_lineage = ncbi_client.get_lineage(taxid)
        name = ncbi_client.get_taxid_translator([taxid])[taxid]
        # Parent taxid of current
        parent_taxid = taxid_lineage[taxid_lineage.index(taxid) - 1]
        for desc_tax_id in ncbi_client.get_descendant_taxa(parent_taxid, collapse_subspecies=False):
            if n_change == 0:
                break
            if str(ncbi_client.get_taxid_translator([desc_tax_id])[desc_tax_id]).startswith(name):
                updated_nodes.add(desc_tax_id)
                n_change -= 1

    def _pad_nodes(
        self, ncbi_client: NCBITaxa, tree: TreeJSON, selected_nodes: list[str], pad_size: int, collect_subspecies: bool
    ) -> set[TaxID]:
        if collect_subspecies:

            def _collect(t_id):
                self._collect_subspecies(updated_nodes, t_id, ncbi_client, pad_size)

        else:

            def _collect(t_id):
                self._pad_at(updated_nodes, t_id, ncbi_client, bd_ids, pad_size)

        updated_nodes: set[TaxID] = set()
        bd_ids: set[TaxID] = self.bacdive_searcher.taxids_with_traits
        current_tx = ncbi_client.get_name_translator(tree["metadata"].keys())
        for tax_name, taxids in current_tx.items():
            updated_nodes.update(taxids)
            if tax_name in selected_nodes:
                for taxid in taxids:
                    _collect(taxid)
        return updated_nodes

    @staticmethod
    @Render.with_update(1, "before")
    def _update_radius(value: int) -> int:
        return value

    # pylint: disable=fixme
    # TODO: Paging on tree
    @Render.with_update(1, "before")
    def _visualize_selection(
        self,
        data: TreeResultData,
        _add: int,
        _prune: int,
        _strain: int,
        selected_nodes: list[str],
        chatbot_provided: list[ChatbotPayload],
    ) -> tuple[str, BacdiveMetadata, list[str]]:
        tree: TreeJSON = data["tree"]
        if ctx.triggered_id == "validated-results":
            return tree["newick"], tree["metadata"], tree["selected_nodes"]

        if ctx.triggered_id == "prune-selection-button":
            if data["tree"]["newick"] == "":
                return no_update, no_update, no_update
            ncbi_client = get_ncbi_taxa()
            name_tx = ncbi_client.get_name_translator(data["tree"]["metadata"].keys())
            pruned_tree = get_ncbi_taxa().get_topology(
                (name_tx[d][0] for d in data["tree"]["metadata"].keys() if d not in selected_nodes),
                intermediate_nodes=True,
            )
            results = TreeUtils.SearchResult(pruned_tree, [], self.bacdive_searcher, []).to_data(chatbot_provided)[
                "tree"
            ]
            return results["newick"], results["metadata"], results["selected_nodes"]

        if ctx.triggered_id == "add-selection-button":
            if data["tree"]["newick"] == "":
                return no_update, no_update, no_update
            pad_size: int = 10
            collect_subspecies = False
        elif ctx.triggered_id == "add-strain-selection-button":
            if data["tree"]["newick"] == "":
                return no_update, no_update, no_update
            pad_size = 10
            collect_subspecies = True
        else:
            raise RuntimeError("unhandled input in TreeRegion._visualize_selection")

        ncbi_client = get_ncbi_taxa()
        padded_tree = ncbi_client.get_topology(
            self._pad_nodes(ncbi_client, tree, selected_nodes, pad_size, collect_subspecies),
            intermediate_nodes=True,
            collapse_subspecies=not collect_subspecies,
        )
        results = TreeUtils.SearchResult(padded_tree, [], self.bacdive_searcher, selected_nodes).to_data(
            chatbot_provided
        )["tree"]
        return results["newick"], results["metadata"], results["selected_nodes"]
