"""Logic for working with "dynamically typed" classes of render-able dash objects"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, LiteralString

from dash import Dash
from dash.development.base_component import Component
from paperqa.docs import Docs

from vaast_app.utils.bacdive_utils import BacdiveAPISearcher
from vaast_app.utils.genetic_tool_wrapper import GeneticToolDBList


# pylint: disable=too-few-public-methods
class Render(ABC):
    """ABC for composite Dash components"""

    InterfaceType = tuple[LiteralString, LiteralString]

    _V: "Render | None" = None

    class Interface:
        """Interface to publicly available subcomponent props"""

    def __new__(cls, *_args, **_kwargs):
        if cls._V is None:
            cls._V = super().__new__(cls)
        return cls._V

    def __init__(
        self,
        parent: "Dash | Render",
        docs: Path | None = None,
        bacdive_searcher: BacdiveAPISearcher | None = None,
        genetic_tools: GeneticToolDBList | None = None,
    ):
        """
        Initialize Render ABC

        :param parent: Initialized Dash app instance or parent to this object
        :param bacdive_searcher: Initialized BacDive searcher. This can be omitted if `parent` is a subclass of Render
        """
        if not hasattr(self, "_initialized"):
            self._initialized = True  # Mark the instance as initialized
            if issubclass(parent.__class__, Render):
                self._app: Dash = parent._app
                self._docs: Docs = parent._docs
                self._bacdive_searcher: BacdiveAPISearcher = parent._bacdive_searcher
                self._genetic_tools: GeneticToolDBList = parent._genetic_tools
            elif isinstance(parent, Dash):
                self._app = parent
                if docs and docs.exists():
                    with open(docs, "rb") as pkl_ptr:
                        self._docs = pickle.load(pkl_ptr)
                else:
                    self._docs = None

                if bacdive_searcher is None:
                    raise ValueError("Top-level 'Render' subclass must provide 'BacdiveAPISearcher' instance")
                if genetic_tools is None:
                    raise ValueError("Top-level 'Render' subclass must provide 'GeneticToolList' instance")
                self._bacdive_searcher = bacdive_searcher
                self._genetic_tools = genetic_tools
            else:
                raise TypeError(f"invalid parent type {type(parent)}")

    @staticmethod
    def with_update(n_components: int, placement: Literal["before", "after"] = "before"):
        """
        Define a function that outputs a list of `n` empty strings before or after its output

        This is meant to aid in use cases that update dcc.Loading.children states

        :param n_components: Number of "" to return
        :param placement: "before" or "after" the output that is returned by the caller
        :return: Decorated function
        """
        if n_components < 0:
            raise ValueError("`n_components` must be non-negative")
        if placement not in ("before", "after"):
            raise ValueError("invalid `placement` provided")

        def fxn(caller: Callable):
            def _f(*args, **kwargs):
                output = caller(*args, **kwargs)
                if not isinstance(output, tuple):
                    output = (output,)
                if placement == "before":
                    return *("" for _ in range(n_components)), *output
                return *output, *("" for _ in range(n_components))

            return _f

        return fxn

    @property
    def app(self) -> Dash:
        """Live app instance"""
        return self._app

    @property
    def bacdive_searcher(self) -> BacdiveAPISearcher:
        """BacdiveAPISearcher instance"""
        return self._bacdive_searcher

    @property
    def genetic_tools(self) -> GeneticToolDBList:
        """GeneticToolList instance"""
        return self._genetic_tools

    @property
    def docs(self) -> Docs:
        """Docs instance"""
        return self._docs

    def __call__(self) -> Component:
        """Generate Component that forms this object"""
        layout = self._set_layout()
        if not issubclass(layout.__class__, Component):
            raise ValueError(
                f"{self.__class__} does not define a valid layout, " f"check `{self.__class__}._set_layout` definition"
            )
        return layout

    @abstractmethod
    def _set_layout(self) -> Component:
        """
        Define `Component` object that will render on `self.__call__`

        :return: `Component` subclass
        """

    def reload_docs(self, path: Path):
        """
        Reload docs from path
        """
        if issubclass(self._app.__class__, Render):
            self._app.reload_docs(path)
            self._docs = self._app.docs
            return

        if not path.exists():
            raise ValueError("Docs file does not exist")
        with open(path, "rb") as pkl_ptr:
            self._docs = pickle.load(pkl_ptr)
