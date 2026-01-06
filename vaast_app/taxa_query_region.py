"""Provides offcanvas component and callback functionality to build queries with AI assistance"""

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html
from dash.development.base_component import Component

from vaast_app.render import Render
from vaast_app.utils.search_by_taxa import set_search_options


# pylint: disable=too-few-public-methods
class TaxaQueryRegion(Render):
    """Component containing query region - offcanvas, query builder, and chatbot functionality"""

    class Interface:
        """Public interface to taxa query region"""

        query: Render.InterfaceType = ("taxa-query-current-selection", "children")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set query based on store data
        self.app.clientside_callback(
            """
            function(queryList) {
                // TODO: Determine source of this added undefined value
                const query = queryList.query.filter(v => v !== undefined);
                console.log(query);
                if (!query) {
                    return [dash_clientside.no_update, dash_clientside.no_update];
                }
                const m = query.map((value, index) => {
                    return {
                        "type": "ListGroupItem", 
                        "namespace": "dash_bootstrap_components", 
                        "props": {"children": value, 
                                  "n_clicks": null, 
                                  "action": true, 
                                  "id": {"type": "query-values", 
                                         "index": index}}
                    };
                });
                return [[...m], null];
            }""",
            Output(*self.Interface.query),
            Output("dropdown-taxa-vis-user-search", "value"),
            Input("store-main-query", "data"),
        )

        # Add value to query store
        self.app.clientside_callback(
            """
            function(_, inputValue, data) {
                data.query.map(v => {
                    if (v === inputValue) {
                        return dash_clientside.no_update;
                    }
                });
                data.query.push(inputValue);
                return data;
            }
            """,
            Output("store-main-query", "data", allow_duplicate=True),
            Input("taxa-query-add", "n_clicks"),
            State("dropdown-taxa-vis-user-search", "value"),
            State("store-main-query", "data"),
            prevent_initial_call=True,
        )

        # Remove value from query store
        self.app.clientside_callback(
            """
            function(n_clicks, data, inputValue) {
                const triggeredId = dash_clientside.callback_context.triggered_id.index;
                // Callback is called when items are first added.
                // This check prevents them from being immediately deleted.
                if (!n_clicks[triggeredId] || !n_clicks) { return [window.dash_clientside.no_update, false, false]; }


                console.log(data.query);

                // TODO: +1 due to presence of undefined value in store
                if (!data.query[0]) {
                    data.query.splice(triggeredId + 1, 1);
                    
                } else {
                    data.query.splice(triggeredId, 1);
                }
                // New list, close tooltip, and disable add button based on what value is currently present
                return [data, false, data.query.some(v => v === inputValue)];
            }
            """,
            Output("store-main-query", "data", allow_duplicate=True),
            Output("query-tooltip", "is_open"),
            Output("taxa-query-add", "disabled", allow_duplicate=True),
            Input({"type": "query-values", "index": ALL}, "n_clicks"),
            State("store-main-query", "data"),
            State("dropdown-taxa-vis-user-search", "value"),
            prevent_initial_call=True,
        )

        # Disable query add button if text field is empty or if value already in query
        self.app.clientside_callback(
            """
            function(inputValue, currentSelection) {
                if (!inputValue || inputValue.length == 0) { return true; }
                for (const entry of currentSelection.query) {
                    if (entry === inputValue) {
                        return true;
                    }
                }
                return false;
            }""",
            Output("taxa-query-add", "disabled", allow_duplicate=True),
            Input("dropdown-taxa-vis-user-search", "value"),
            State("store-main-query", "data"),
            prevent_initial_call=True,
        )

        self.app.callback(
            Output("dropdown-taxa-vis-user-search", "options"),
            Input("dropdown-taxa-vis-user-search", "search_value"),
            prevent_initial_call=True,
        )(set_search_options)

    @staticmethod
    def _user_input_region() -> Component:
        return dbc.InputGroup(
            [
                dbc.Container(
                    [
                        dbc.Row(
                            html.Div(
                                dcc.Dropdown(
                                    options=[],
                                    placeholder="Enter species...",
                                    className="container-fluid rounded multi-line-dropdown",
                                    id="dropdown-taxa-vis-user-search",
                                    style={"whiteSpace": "normal", "lineHeight": "15px"},
                                ),
                            ),
                            class_name="mb-2",
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    dbc.Button(
                                        "Add to query",
                                        id="taxa-query-add",
                                        disabled=True,
                                    ),
                                    className="d-flex justify-content-center",
                                ),
                                className="d-flex justify-content-center",
                            )
                        ),
                    ],
                ),
            ]
        )

    def _current_query_region(self) -> Component:
        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.ListGroup(
                                [],
                                id=self.Interface.query[0],
                            ),
                            dbc.Tooltip(
                                "Click to remove",
                                target=self.Interface.query[0],
                                delay={"show": 0, "hide": 5},
                                id="query-tooltip",
                            ),
                        ]
                    ),
                    class_name="mb-2",
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.ListGroup(id="chatbot-data-vis"),
                            dbc.Tooltip(
                                "Click to remove",
                                target="chatbot-data-vis",
                                delay={"show": 0, "hide": 5},
                                id="chatbot-tooltip",
                            ),
                        ]
                    ),
                    class_name="mb-2",
                ),
            ],
        )

    def _set_layout(self) -> Component:
        return dbc.Container(
            [
                dbc.Row(dbc.Col(html.Div("Select bacteria for chat", className="h3")), class_name="mb-2"),
                dbc.Row(dbc.Col(self._user_input_region()), class_name="mb-2"),
                dbc.Row(dbc.Col(self._current_query_region())),
            ]
        )


if __name__ == "__main__":
    ...
