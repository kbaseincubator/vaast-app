"""Chatbot Region"""

import json
from typing import Literal

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html
from dash.dash_table import DataTable
from dash.development.base_component import Component

from vaast_app.render import Render
from vaast_app.utils.chatbot.chat_logger import ChatLogger, TokenPayload
from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload, ChatbotUtils, ChatMessage, ChatMessageList

button_style = {"background-color": "#5E9732", "border": "#5E9732"}


# pylint: disable=too-few-public-methods
class ChatbotRegion(Render):
    """Chatbot Region"""

    class Interface:
        """Interface to Chatbot"""

        # Parsed bacteria/tool information
        data: Render.InterfaceType = ("chatbot-provided", "data")
        # Conversation with user, including rating and sender
        conversation: Render.InterfaceType = ("store-conversation", "data")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chatbot = ChatbotUtils(docs=self.docs)
        self._chat_logger = ChatLogger()

        self.app.callback(
            Output("display-conversation", "children"),
            Input(*self.Interface.conversation),
            Input("chatbot-submit-button", "n_clicks"),
            prevent_initial_call=False,
        )(self._update_display)

        self.app.callback(
            Output("chatbot-submit-button", "disabled", allow_duplicate=True),
            Output("chatbot-visualize-button", "disabled", allow_duplicate=True),
            Output("chatbot-reset-button", "disabled", allow_duplicate=True),
            Output("user-input", "value", allow_duplicate=True),
            Input("chatbot-submit-button", "n_clicks"),
            prevent_initial_call=True,
        )(self._disable_submit_and_clear_input)

        self.app.callback(
            Output(*self.Interface.conversation, allow_duplicate=True),
            Output("chatbot-visualize-button", "disabled", allow_duplicate=True),
            Output("user-input", "value", allow_duplicate=True),
            Output("last-selected", "data", allow_duplicate=True),
            Output("thumbs-up-btn", "active", allow_duplicate=True),
            Output("thumbs-down-btn", "active", allow_duplicate=True),
            Input("chatbot-reset-button", "n_clicks"),
            State("chat-token", "data"),
            State(*self.Interface.conversation),
            prevent_initial_call=True,
        )(self._reset_conversation)

        self.app.callback(
            Output(*self.Interface.conversation, allow_duplicate=True),
            Output("chatbot-submit-button", "disabled", allow_duplicate=True),
            Output("chatbot-visualize-button", "disabled", allow_duplicate=True),
            Output("chatbot-reset-button", "disabled", allow_duplicate=True),
            Output("chatbot-thinking-loading", "children", allow_duplicate=True),
            Output("thumbs-up-btn", "active", allow_duplicate=True),
            Output("thumbs-down-btn", "active", allow_duplicate=True),
            Input("chatbot-submit-button", "n_clicks"),
            Input("user-input", "n_submit"),
            State("user-input", "value"),
            State(*self.Interface.conversation),
            State("chat-token", "data"),
            State("store-main-query", "data"),
            State("results-tree", "selected_nodes"),
            prevent_initial_call=True,
            # background=True,
        )(self._run_chatbot)

        self.app.callback(
            Output(*self.Interface.data, allow_duplicate=True),
            Output("chat-error-modal", "is_open"),
            Output("chatbot-thinking-loading", "children", allow_duplicate=True),
            Input("chatbot-visualize-button", "n_clicks"),
            State(*self.Interface.conversation),
            State(*self.Interface.data),
            prevent_initial_call=True,
            # background=True,
        )(self._visualize_chatbot_data)

        self.app.callback(
            Output("chat-token", "data"), Input("chatbot-reset-button", "n_clicks"), prevent_initial_call=False
        )(self._generate_token)

        self.app.callback(
            Output("ratings-div", "style"), Input("store-conversation", "data"), prevent_initial_call=False
        )(self._hide_rating_region)

        self.app.callback(
            Output(*self.Interface.conversation),
            Output("thumbs-up-btn", "active", allow_duplicate=True),
            Output("thumbs-down-btn", "active", allow_duplicate=True),
            Output("last-selected", "data", allow_duplicate=True),
            Input("thumbs-up-btn", "n_clicks"),
            Input("thumbs-down-btn", "n_clicks"),
            State("store-conversation", "data"),
            State("chat-token", "data"),
            State("last-selected", "data"),
            prevent_initial_call=True,
        )(self._rate_last_response)

    def _set_layout(self) -> Component:
        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            html.H4("Genetic Tools ChatBot"),
                            dcc.Store(
                                id=self.Interface.conversation[0],
                                data=self._chatbot.initialize_chat_history(),
                            ),
                            dcc.Store(id="chat-token", data={"token": None}),
                            dcc.Store(id="chatbot-provided", data=[]),
                            dcc.Store(id="last-selected", data=None),
                        ]
                    )
                ),
                html.Hr(),
                # pylint: disable=fixme
                # TODO: Interactivity in chat response
                #  - Highlight organisms that will be visualized
                #  - Allow users to de/select what will be visualized
                dbc.Row(
                    dbc.Col(
                        [
                            html.Div(
                                # html.Div(id="display-conversation"),
                                style={
                                    "overflowY": "auto",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "height": "250px",
                                    "scroll-margin-bottom": "5px",
                                    "scroll-snap-align": "end none",
                                },
                                id="display-conversation",
                            ),
                            html.Div(["Chatting with: ", html.Span(id="chatting-with-text")]),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Error")),
                                    dbc.ModalBody(
                                        "Unable to add data to visualizer. Please try again, or ask a different question."
                                    ),
                                ],
                                id="chat-error-modal",
                                is_open=False,
                            ),
                        ]
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    "Was this response useful?",
                                    dbc.Button(
                                        html.I(className="bi bi-hand-thumbs-up"),
                                        id="thumbs-up-btn",
                                        className="btn btn-light ms-2 p-1",
                                    ),
                                    dbc.Button(
                                        html.I(className="bi bi-hand-thumbs-down"),
                                        id="thumbs-down-btn",
                                        className="btn btn-light ms-2 p-1",
                                    ),
                                ],
                                className="g-0",
                                id="ratings-div",
                            )
                        ]
                    ),
                    className="mb-1 ms-1",
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            # what phages can be engineered into pseudomonas aeruginosa?
                            html.Div(
                                children=[
                                    html.Div(
                                        dbc.Textarea(
                                            id="user-input",
                                            rows=5,
                                            placeholder="Write to the chatbot...",
                                        )
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Ask",
                                                id="chatbot-submit-button",
                                            ),
                                            dbc.Button(
                                                "View on Tree",
                                                id="chatbot-visualize-button",
                                                disabled=True,
                                            ),
                                            dbc.Button(
                                                "Reset",
                                                id="chatbot-reset-button",
                                            ),
                                        ],
                                        className="d-flex justify-content-center",
                                    ),
                                ]
                            ),
                            dcc.Loading(html.Div(id="chatbot-thinking-loading")),
                        ]
                    ),
                    className="pb-4",
                ),
            ],
            class_name="border",
            fluid=True,
        )

    @staticmethod
    def _generate_token(_: int) -> TokenPayload:
        return ChatLogger.generate_chat_token()

    @staticmethod
    def _textbox(text: str, message_type: str, box: Literal["AI", "user"] = "AI") -> dbc.Card:
        style: dict[str, str | int] = {
            "maxWidth": "60%",
            "width": "max-content",
            "padding": "5px 10px",
            "borderRadius": 25,
            "marginBottom": 20,
            "whiteSpace": "pre-wrap",
        }

        if box == "user":
            style["marginLeft"] = "auto"
            style["marginRight"] = 0
            style["border"] = "#275625"

            return dbc.Card(
                dcc.Markdown(text), style=style, body=True, inverse=True, className="shadow-sm", color="#5E9732"
            )

        if box == "AI":
            style["marginLeft"] = 0
            style["marginRight"] = "auto"

            if message_type == "text":
                return dbc.Card(
                    dcc.Markdown(text), style=style, body=True, color="light", inverse=False, className="shadow-sm"
                )
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return dbc.Card(
                    dcc.Markdown(text), style=style, body=True, color="light", inverse=False, className="shadow-sm"
                )
            return dbc.Card(
                DataTable(data, page_size=8, sort_action="native", page_action="native"),
                style=style,
                body=True,
                color="light",
                inverse=False,
                className="shadow-sm",
            )
        raise ValueError("incorrect option for `box`")

    @staticmethod
    def _update_display(chat_history: list[ChatMessage], _: int) -> list[Component]:
        return [
            ChatbotRegion._textbox(x["message"], x["message_type"], box="user" if i % 2 == 1 else "AI")
            for i, x in enumerate(chat_history)
        ]

    @staticmethod
    def _disable_submit_and_clear_input(_: int) -> tuple[bool, bool, bool, None]:
        return True, True, True, None

    def _reset_conversation(
        self, _: int, chat_token: TokenPayload, chat_history: ChatMessageList
    ) -> tuple[ChatMessageList, bool, None, None, bool, bool]:
        self._chat_logger.log(chat_token, chat_history)
        return self._chatbot.initialize_chat_history(), True, None, None, False, False

    @Render.with_update(1, "after")
    def _visualize_chatbot_data(
        self, _: int, chat_history: ChatMessageList, existing_data: list[ChatbotPayload]
    ) -> tuple[list[ChatbotPayload], bool]:
        try:
            return existing_data + [self._chatbot.get_visualization_data(chat_history)], False
        except RuntimeError:
            return existing_data, True

    @Render.with_update(1, "after")
    def _run_chatbot(
        self,
        n_clicks: int,
        n_submit: int,
        user_input: str,
        chat_history: ChatMessageList,
        chat_token: TokenPayload,
        main_store: dict[str, list[str]],
        tree_selection: list[str],
    ) -> tuple[ChatMessageList, bool, bool, bool, bool, bool]:
        if n_clicks == 0 and n_submit is None or (user_input is None or user_input == ""):
            return chat_history, False, len(chat_history) == 1, False, False, False

        self._chatbot.chat(user_input, chat_history, main_store, tree_selection)
        self._chat_logger.log(chat_token, chat_history)

        return chat_history, False, False, False, False, False

    @staticmethod
    def _hide_rating_region(chat_history: ChatMessageList) -> dict[str, str]:
        if len(chat_history) <= 1:
            return {"display": "none"}
        return {"display": "inline"}

    def _rate_last_response(
        self, _n1: int, _n2: int, chat_history: ChatMessageList, chat_token: TokenPayload, last_selected: str | None
    ) -> tuple[ChatMessageList, bool, bool, str]:
        existing_rating = chat_history[-1]["rating"]
        triggered_id = ctx.triggered_id
        if ctx.triggered_id == "thumbs-up-btn":
            if existing_rating == "good":
                rating = None
            else:
                rating = "good"
            if last_selected == ctx.triggered_id:
                b1_active, b2_active = False, False
                triggered_id = None
            else:
                b1_active, b2_active = True, False
        elif ctx.triggered_id == "thumbs-down-btn":
            if existing_rating == "bad":
                rating = None
            else:
                rating = "bad"
            if last_selected == ctx.triggered_id:
                b1_active, b2_active = False, False
                triggered_id = None
            else:
                b1_active, b2_active = False, True
        else:
            raise ValueError(f"invalid triggered_id `{ctx.triggered_id}`")

        self._chat_logger.update_rating(rating, chat_history)
        self._chat_logger.log(chat_token, chat_history)

        return chat_history, b1_active, b2_active, triggered_id
