"""Chat logging utilities"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import NewType, TypedDict, cast

from vaast_app.utils.chatbot.chatbot_utils import ChatMessage

ChatToken = NewType("ChatToken", str)


class TokenPayload(TypedDict):
    """Token payload"""

    token: ChatToken


class ChatLogger:
    """
    Logger class for storing chat convesations to disk
    """

    output_dir: Path

    def __init__(self):
        """Create logger"""
        self._validate()

    @staticmethod
    def set_output_dir(path: Path) -> None:
        """
        Output directory for chat logs

        Must be set prior to creating a `ChatLogger` instance

        :param path: Output path
        :return: None
        """
        path.mkdir(parents=True, exist_ok=True)
        ChatLogger.output_dir = path

    @staticmethod
    def _validate() -> None:
        """Confirm that output directory has been set"""
        if ChatLogger.output_dir is None:
            raise ValueError("output directory not set")
        if not ChatLogger.output_dir.exists():
            ChatLogger.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_chat_token() -> TokenPayload:
        """
        Create token for chat

        :return: Token
        """
        ChatLogger._validate()
        return {"token": ChatToken(str(uuid.uuid4()))}

    @staticmethod
    def update_rating(rating: str, chat_history: list[ChatMessage]) -> None:
        """
        Update rating of latest response from AI

        :param rating: New rating
        :param chat_history: History of chat
        :return: None
        """
        chat_history[-1]["rating"] = rating

    def log(self, chat_token: TokenPayload, chat_history: list[ChatMessage]) -> None:
        """
        Store current chat history using generated token

        :param chat_token: Token for retrieval/storage
        :param chat_history: History of chat
        :return: None
        """
        self._validate()
        with open(cast(Path, self.output_dir).joinpath(f"{chat_token['token']}-chat-log.json"), "w") as json_ptr:
            json.dump({"chat-history": chat_history, "datetime": str(datetime.now())}, json_ptr, indent=2)
