"""
Tests for the chatbot_calls module.
"""

import pickle

import pytest
from pydantic import BaseModel
from pydantic.fields import Field

from vaast_app.utils.chatbot.chatbot_calls import ChatbotClient, ChatMessage, ChatRequest


@pytest.fixture(scope="module")
def chatbot_client():
    """
    Fixture to create a ChatbotClient instance for testing.
    """
    with open("data/openai-docs-all.pkl", "rb") as pkl_ptr:
        docs = pickle.load(pkl_ptr)
    return ChatbotClient(
        docs=docs,
        version="OpenAI",
        use_cborg=False,
        model="gpt-5.1",
    )


class _Response(BaseModel):
    """
    Class handles model response that checks if a chatbot response is adequate for the user question
    """

    is_adequate: bool = Field(description="Whether the chatbot response is adequate for the user question")


def _test_response(client: ChatbotClient, user_question: str, chatbot_response: str) -> bool:
    """
    Helper function to check if the chatbot response is adequate for the user's question.
    """
    response = client.generate_response(
        prompt=(
            f"A user has asked the following question: {user_question}\n"
            f"Is the following text a valid response to the user question?\n"
            f"TEXT RESPONSE: {chatbot_response}\n"
        ),
        return_type=_Response,
    )
    return response.is_adequate


def test_chatbot_conversation_flow(chatbot_client: ChatbotClient):
    """
    Test the conversation flow of the chatbot to ensure basic functionality and context management.
    """
    # First request
    chat_request_1 = ChatRequest(
        message="Tell me about plasmids that confer kanomycin resistance.",
        chat_history=[],
        selection=["Pseudomonas aeruginosa"],
        message_type="text",
    )
    response_1 = chatbot_client.process_chat(chat_request_1)
    assert _test_response(chatbot_client, chat_request_1.message, response_1.message)

    # Second request (continuation)
    chat_request_2 = ChatRequest(
        message="Tell me more about chromosomal tagging.",
        chat_history=[
            ChatMessage(
                sender="user",
                message=chat_request_1.message,
                message_type=chat_request_1.message_type,
                selection=chat_request_1.selection,
                rating=None,
            ),
            ChatMessage(
                sender="system",
                message=response_1.message,
                message_type=response_1.message_type,
                selection=response_1.selection,
                rating=None,
            ),
        ],
        selection=chat_request_1.selection,
        message_type=chat_request_1.message_type,
    )
    response_2 = chatbot_client.process_chat(chat_request_2)
    print(response_2.message)
    assert _test_response(chatbot_client, chat_request_2.message, response_2.message)
