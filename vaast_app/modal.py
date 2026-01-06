from logging import getLogger

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, no_update
from dash.development.base_component import Component

from vaast_app.render import Render
from vaast_app.taxa_query_region import TaxaQueryRegion
from vaast_app.tree_region import TreeRegion
from vaast_app.utils.tree_utils import TreeResultData


class ModalRegion(Render):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = getLogger("modal")
        self._taxa_query_region: TaxaQueryRegion = TaxaQueryRegion(self)

        # Take NCBI-tax-validated results and provide to visualizer callback
        #   TreeRegion._visualize_selection
        self.app.callback(
            Output("validated-results", "data"),
            Output("error-modal", "is_open"),
            Output("error-modal-body", "children"),
            Input("ncbi-query-results", "data"),
            prevent_initial_call=True,
        )(self._error_check_ncbi_query)

        self.app.clientside_callback(
            """
            function(n_clicks) {
                return false;
            }
            """,
            Output("selection-modal", "is_open", allow_duplicate=True),
            Input("btn-selection-modal-close", "n_clicks"),
            prevent_initial_call=True,
        )

    def _set_layout(self) -> Component:
        return dbc.Modal(
            children=[
                dbc.Row(
                    [
                        dbc.Col(self._taxa_query_region(), className="col-4 mt-4"),
                        dbc.Col(
                            [
                                dbc.Container(
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Button(
                                                html.I(className="fa-solid fa-backward"),
                                                id="btn-selection-modal-close",
                                            ),
                                            className="ms-auto col-2",
                                        ),
                                        className="d-flex align-items-right",
                                    ),
                                    fluid=True,
                                    className="mb-4",
                                ),
                                dcc.Loading(id="tree-loading", fullscreen=True, type="circle"),
                                dcc.Loading(id="strain-stats-loading", type="circle"),
                                dcc.Loading(id="genetic-tools-loading", type="circle"),
                                dbc.Container(
                                    dbc.Row(self._generate_visualizer()),
                                    id="visualizer-container",
                                    class_name="g-0",
                                ),
                            ],
                            class_name="g-0 col-8",
                        ),
                    ],
                    class_name="m-1 h-100 oveflow-auto",
                ),
            ],
            id="selection-modal",
            fullscreen=True,
            is_open=False,
        )

    def _generate_visualizer(self) -> list[Component]:
        return [
            dbc.Col(
                [
                    dbc.Container(
                        dbc.Row(dbc.Col(TreeRegion(self)(), class_name="overflow-auto h-100")), class_name="mb-2"
                    )
                ],
                class_name="col-12 col-lg-12",
            ),
        ]

    def _error_check_ncbi_query(self, query_results: TreeResultData) -> tuple[TreeResultData, bool, str]:
        failed = query_results["failed"]
        if len(failed) > 0:
            self._logger.info("Failed items from query: %s", ",".join(failed))
            return query_results, True, f"Query failed for {failed}"
        return query_results, False, no_update
