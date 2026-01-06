"""Utilities for interacting with Chatbot"""

import os
from json import JSONDecodeError
from logging import getLogger
from typing import Literal, TypedDict

from paperqa.docs import Docs

from vaast_app.utils.chatbot.chatbot_calls import ChatbotClient, ChatRequest
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.type_utils import TaxName


class ModelGeneticTool(TypedDict):
    """Description of genetic tool"""

    type: str
    name: str
    references: list[str]


class ModelResponse(TypedDict):
    """Species and the relevant tools describing it"""

    species: TaxName
    tools: list[ModelGeneticTool]


class ChatbotPayload(TypedDict):
    """Parsed output from Chatbot"""

    results: list[ModelResponse]


class ChatMessage(TypedDict):
    """Message stored by Chatbot"""

    message: str
    sender: str
    selection: list[str]
    message_type: Literal["text", "table"]
    rating: str | None


class ChatMessageList(list[ChatMessage]):
    """Wrapper class for list of `ChatMessage` objects"""

    def __init__(self, *args):
        """Initialize the ChatMessageList."""
        super().__init__(*args)


class ChatbotUtils:
    """
    Utility class for interacting with Chatbot
    """

    CHATBOT_ROLE = "system"

    def __init__(self, docs: Docs | None):
        """Initialize the ChatbotUtils."""
        self._logger = getLogger("chatbot-utils")
        self._stop_text = "STOP"
        self._docs = docs

        version = os.environ.get("VERSION")
        if not version:
            if docs and "anthropic" in docs.name.lower():
                version = "Anthropic"
            else:
                version = "OpenAI"

        self._chat_client = ChatbotClient(
            docs,
            version,
            os.environ.get("HOSTING_LOCATION", "API"),
            os.environ.get("MODEL", "gpt-5.1"),
        )

    @staticmethod
    def initialize_chat_history() -> ChatMessageList:
        """Create initial message"""
        return ChatMessageList(
            [
                {
                    "message": "Hi! What questions do you have today?",
                    "message_type": "text",
                    "sender": ChatbotUtils.CHATBOT_ROLE,
                    "rating": None,
                }
            ]
        )

    def _chat(
        self,
        user_input: str,
        chat_history: list[ChatMessage],
        modify_in_place: bool,
        selection: list[str] | None,
    ) -> ChatMessage:
        if selection is None:
            selection = []
        self._logger.info("chat: [bacteria: %s\n%s\n]", ",".join(selection), user_input)

        if self._docs is None:
            response = {
                "message": "Documentation not loaded. Please select a Version in the Settings menu.",
                "sender": ChatbotUtils.CHATBOT_ROLE,
                "rating": None,
                "message_type": "text",
                "selection": selection,
            }
            if modify_in_place:
                chat_history.append(
                    {
                        "message": user_input,
                        "sender": "user",
                        "rating": None,
                        "selection": selection,
                        "message_type": "text",
                    }
                )
                chat_history.append(response)
            return response

        try:
            response = self._chat_client.process_chat(
                ChatRequest(
                    message=user_input,
                    chat_history=chat_history[1:],
                    selection=selection,
                    message_type="text",
                ),
            ).model_dump()
        except JSONDecodeError:
            return {
                "message": "Server error.",
                "sender": ChatbotUtils.CHATBOT_ROLE,
                "rating": None,
                "message_type": "text",
                "selection": selection,
            }
        self._logger.info("LLM response: [\n%s\n]", response["message"])
        response: ChatMessage = {
            "message": str(response["message"]),
            "message_type": response["message_type"],
            "sender": ChatbotUtils.CHATBOT_ROLE,
            "rating": None,
            "selection": selection,
        }
        if modify_in_place:
            chat_history.append(
                {
                    "message": user_input,
                    "sender": "user",
                    "rating": None,
                    "selection": selection,
                    "message_type": "text",
                }
            )
            chat_history.append(response)
        return response

    def chat(
        self,
        user_input: str,
        chat_history: list[ChatMessage],
        main_store: dict[str, list[str]],
        tree_selection: list[str],
    ) -> None:
        """
        Chat with Chatbot

        `chat_history` will be modified in place

        :param user_input: Message to Chatbot
        :param chat_history: Message history list
        :return: Response from Chatbot
        """
        self._chat(
            user_input, chat_history, True, [v for v in set(tree_selection) | set(main_store["query"]) if v is not None]
        )

    def get_visualization_data(self, chat_history: list[ChatMessage]) -> ChatbotPayload:
        """
        View biological entities identified in the last message in the `chat_history`

        :param chat_history: Chat history
        :return: Response from payload parsed to `organisms` and `genetic_tools`
        """
        # Try 3 times to collect results
        message_parsed = ""
        ncbi_client = get_ncbi_taxa()
        for i in range(3):
            try:
                message_parsed = self._chat_client.visualizer_data(
                    ChatRequest(
                        message="",
                        chat_history=chat_history[-1:],
                        selection=chat_history[-1]["selection"],
                        message_type="text",
                    ),
                ).model_dump()
                self._logger.info("LLM response: [\n%s\n]", message_parsed)

                return {
                    "results": [
                        {
                            key: (
                                value
                                if key != "species"
                                else next(
                                    iter(
                                        ncbi_client.get_taxid_translator(
                                            ncbi_client.get_name_translator([value])[value]
                                        ).values()
                                    )
                                )
                            )
                            for key, value in entry.items()
                        }
                        for entry in message_parsed
                    ]
                }
            except Exception as err:  # pylint: disable=broad-exception-caught
                self._logger.error("Error on try %i: %s", i, str(err))
                continue
        self._logger.error("Unable to parse model results from message [\n%s\n]", message_parsed)
        raise RuntimeError(f"unable to parse results from model\n\n{message_parsed}")
