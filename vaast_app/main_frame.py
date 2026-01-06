"""Main frame region"""

from logging import getLogger
from pathlib import Path
from string import punctuation

import dash_bootstrap_components as dbc
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html
from dash.development.base_component import Component

from vaast_app.chatbot_region import ChatbotRegion
from vaast_app.genetic_tools_region import GeneticTools
from vaast_app.modal import ModalRegion
from vaast_app.render import Render
from vaast_app.strain_stats import StrainStats
from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload
from vaast_app.utils.tree_utils import TreeResultData, TreeUtils


class MainFrame(Render):
    """Main frame region"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = getLogger("main-frame")
        self.chatbot = ChatbotRegion(self)

        # Visualize chatbot data
        self.app.clientside_callback(
            ClientsideFunction(namespace="chatbot_vis", function_name="add_chatbot_data"),
            Output("chatbot-data-vis", "children", allow_duplicate=True),
            Input(*self.chatbot.Interface.data),
            prevent_initial_call=True,
        )
        # Take user- and chatbot-provided data and validate against NCBI taxonomy
        self.app.callback(
            Output("tree-loading", "children", allow_duplicate=True),
            Output("ncbi-query-results", "data"),
            Input("store-main-query", "data"),
            Input(*self.chatbot.Interface.data),
            prevent_initial_call=True,
            # background=True,  # TODO: Check if this is necessary - do queries last > 30 sec?
        )(self._query_ncbi_with_user_and_chatbot_data)
        # Remove entries from chatbot-provided
        self.app.clientside_callback(
            """
            function(n_clicks, chatbotVisData, chatbotData) {
                const triggeredId = dash_clientside.callback_context.triggered_id;
                const noUpdate = window.dash_clientside.no_update;
                if (triggeredId === "chatbot-provided") { return [noUpdate, noUpdate, false]; }
                // Callback is called when items are first added. 
                // This check prevents them from being immediately deleted.
                if (!n_clicks[triggeredId.index]) { return [noUpdate, noUpdate, false]; }
                chatbotVisData.splice(triggeredId, 1);
                // Re-number elements for future callbacks
                chatbotVisData.map((elem, i) => elem.props.id.index = i);
                // Remove entry from main data store
                chatbotData.splice(triggeredId, 1);
                // New list, close tooltip
                return [chatbotVisData, chatbotData, false];
            }
            """,
            Output("chatbot-data-vis", "children", allow_duplicate=True),
            Output(*self.chatbot.Interface.data, allow_duplicate=True),
            Output("chatbot-tooltip", "is_open"),
            Input({"type": "chatbot-values", "index": ALL}, "n_clicks"),
            State("chatbot-data-vis", "children"),
            State(*self.chatbot.Interface.data),
            prevent_initial_call=True,
        )
        # Open selection modal
        self.app.clientside_callback(
            """
            function(n_clicks) {
                return true;
            }  
            """,
            Output("selection-modal", "is_open", allow_duplicate=True),
            Input("btn-open-selection-modal", "n_clicks"),
            prevent_initial_call=True,
        )

        self.app.clientside_callback(
            """
            function(selected_nodes) {
                if (selected_nodes.length === 0) {
                    return "all organisms";
                }
                if (selected_nodes.length > 3) {
                    return selected_nodes.slice(0, 3).join(", ") + ", " + `(+${selected_nodes.length - 3} more)`;
                }
                return selected_nodes.join(", ");
            }
            """,
            Output("chatting-with-text", "children"),
            Input("results-tree", "selected_nodes"),
            prevent_initial_call=True,
        )

    def _set_layout(self) -> Component:
        return dbc.Container(
            [
                dcc.Store(id="validated-results", data={}),
                dcc.Store(
                    id="ncbi-query-results",
                    data={},
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Error occurred"),
                        dbc.ModalBody(id="error-modal-body"),
                        dbc.ModalFooter(),
                    ],
                    id="error-modal",
                    is_open=False,
                    size="sm",
                ),
                ModalRegion(self)(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.A(
                                    html.Img(
                                        src="https://www.kbase.us/wp-content/uploads/sites/6/2020/08/k-base_logo.svg",
                                        style={"width": "250px"},
                                        className="pt-2 p-1",
                                    ),
                                    href="/",
                                ),
                            ],
                        ),
                    ],
                    class_name="d-flex",
                ),
                dbc.Container(
                    dbc.Row(
                        dbc.Col(
                            dbc.Button("Select bacteria", id="btn-open-selection-modal"), className="ms-auto col-2"
                        ),
                        className="d-flex align-items-right",
                    ),
                    fluid=True,
                    className="mb-4",
                ),
                dbc.Container(
                    [self.chatbot()],
                    class_name="g-0",
                    fluid=True,
                    className="mt-2",
                ),
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                StrainStats(self)(),
                                GeneticTools(self)(),
                            ],
                            class_name="align-items-end",
                        )
                    ],
                    class_name="g-0",
                    fluid=True,
                ),
            ],
            class_name="border rounded border-dark ms-0 me-0 mt-1",
            fluid=True,
        )

    @Render.with_update(1, "before")
    def _query_ncbi_with_user_and_chatbot_data(
        self,
        data: dict[str, list[str]],
        chatbot_provided: list[ChatbotPayload],
    ) -> TreeResultData:
        # TODO: Determine source of this added undefined value
        query = [v for v in data["query"] if v is not None]
        if chatbot_provided is not None:
            for result_dict in chatbot_provided:
                for result in result_dict["results"]:
                    if result["species"] not in query:
                        query.append(result["species"])
        tree_utils = TreeUtils(self.bacdive_searcher, Path("data/ncbi-tax-names.parquet"))
        return tree_utils.bacdive_based_search([q.strip(punctuation) for q in query]).to_data(chatbot_provided or None)
