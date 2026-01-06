import os
from pathlib import Path

from dash import get_app, html, register_page

from vaast_app.main_frame import MainFrame
from vaast_app.utils.bacdive_utils import BacdiveAPISearcher
from vaast_app.utils.chatbot.chat_logger import ChatLogger
from vaast_app.utils.genetic_tool_wrapper import load_genetic_tools

register_page(__name__, path="/visualizer")
ChatLogger.set_output_dir(Path(os.environ.get("CHAT_LOGS_DIR", "chat-logs")))


tools = load_genetic_tools()
bacdive_searcher = BacdiveAPISearcher.get_searcher()
layout = html.Div(
    [
        html.Link(rel="stylesheet"),
        html.Div(
            MainFrame(get_app(), Path(os.environ.get("DOCS", "unable-to-locate-docs")), bacdive_searcher, tools)(),
            className="d-none d-md-block g-0 mh-100",
        ),
        html.Div(["This page is not optimized for small screens"], className="d-block d-md-none g-0"),
    ],
    className="px-2 py-1",
)
