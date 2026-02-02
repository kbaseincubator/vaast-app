"""
Genetic tools region
"""

from collections import defaultdict
from typing import Iterator, cast

import dash_bootstrap_components as dbc
import polars as pl
from dash import Input, Output, dash_table, html
from dash.development.base_component import Component

from vaast_app.render import Render
from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload, ModelGeneticTool
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.summary_region import SummaryRow, summary_row
from vaast_app.utils.type_utils import TaxName


class GeneticTools(Render):
    """Genetic tools region"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app.callback(
            Output("genetic-tools-container", "children"),
            Output("genetic-tools-loading", "children"),
            Input("results-tree", "selected_nodes"),
            Input("chatbot-provided", "data"),
            prevent_initial_call=True,
        )(self._show_genetic_tools)

    def _set_layout(self) -> Component:
        return dbc.Col(
            id="genetic-tools-col",
            children=[
                html.H4("Genetic tools"),
                dbc.Container(className="g-0 overflow-auto", id="genetic-tools-container", style={"height": "32vh"}),
            ],
            className="border border-secondary pt-1 pb-2 col-6",
        )

    @staticmethod
    def _to_table(values: list[dict[str, str]]) -> dash_table.DataTable:
        return dash_table.DataTable(
            values,
            style_table={"overflowX": "scroll", "textAlign": "left"},
            page_size=10,
            style_header={"textAlign": "left"},
            page_action="native",
            filter_action="native",
            sort_action="native",
            columns=[{"name": col, "id": col, "presentation": "markdown"} for col in list(values[0].keys())],
        )

    def _collect_tooldb_info(self, species: str, collected_tool_dfs: list[pl.DataFrame]) -> list[list[dict[str, str]]]:
        data: list[list[dict[str, str]]] = [[] for _ in range(len(self.genetic_tools))]
        for genetic_tool, c_df, data_ptr in zip(self.genetic_tools, collected_tool_dfs, data, strict=True):
            species_df = c_df.filter(pl.col("species").eq(species)).select(
                genetic_tool.tool_id_col, genetic_tool.tool_name
            )
            for row in species_df.iter_rows(named=True):
                data_ptr.extend(
                    [
                        {"ID": x[0], "Name": x[1]}
                        for x in zip(row[genetic_tool.tool_id_col], row[genetic_tool.tool_name], strict=False)
                    ]
                )
        return data

    def _summary_row_data(
        self,
        species: TaxName,
        alt_species: TaxName,
        data: list[list[dict[str, str]]],
        item_ids: list[str],
        model_provided: dict[TaxName, list[ModelGeneticTool] | str],
    ) -> Iterator[dbc.AccordionItem]:
        for data_ptr, genetic_tool, item_id in zip(data, self.genetic_tools, item_ids, strict=False):
            if len(data_ptr) > 0:
                yield dbc.AccordionItem(
                    GeneticTools._to_table(data_ptr), title=genetic_tool.database_name, item_id=item_id
                )
        model_entry = model_provided.get(species, [])
        if len(model_entry) == 0:
            model_entry = model_provided.get(alt_species, [])
        if len(model_entry) > 0:
            yield dbc.AccordionItem(
                GeneticTools._to_table(cast(list, sorted(model_entry, key=lambda v: (v["type"].lower(), v["type"])))),
                title="Model-generated results",
                item_id="model-generated",
            )

    def _to_summary_rows(
        self,
        bd_df: pl.DataFrame,
        collected_tool_dfs: list[pl.DataFrame],
        model_provided: dict[TaxName, list[ModelGeneticTool] | str],
    ) -> list[SummaryRow]:
        item_ids = ["phd", "plsdb", "vhd"]

        out = []
        bd_df = bd_df.with_columns(
            (
                pl.struct("species-taxid", "species")
                .replace(get_ncbi_taxa().get_taxid_translator(bd_df["species-taxid"]), default=pl.col("species"))
                .alias("species-replaced")
            )
        ).sort("species-replaced")
        found_species = set()
        for species, species_repl, species_taxid in zip(
            bd_df["species"], bd_df["species-replaced"], bd_df["species-taxid"], strict=False
        ):
            if species in found_species:
                continue
            found_species.add(species)
            data = self._collect_tooldb_info(species, collected_tool_dfs)
            if sum(len(v) for v in data) == 0:
                continue
            out.append(
                summary_row(
                    (
                        f"{species_taxid} {species_repl} ({species})"
                        if species != species_repl
                        else f"{species_taxid} {species_repl}"
                    ),
                    self._summary_row_data(species, species_repl, data, item_ids, model_provided),
                )
            )
        return out

    @staticmethod
    def _collapse_to_single_entry(
        chatbot_provided: list[ChatbotPayload] | None,
    ) -> dict[TaxName, list[ModelGeneticTool]]:
        if chatbot_provided is None:
            return {}
        out = defaultdict(list)
        for value in chatbot_provided or []:
            out[value["species"]].extend(cast(ModelGeneticTool, value["tools"]))
        return out

    @Render.with_update(1, "after")
    def _show_genetic_tools(
        self, selected_nodes: list[str], chatbot_provided: list[ChatbotPayload] | None
    ) -> list[SummaryRow]:
        bd_df = self.bacdive_searcher.get_taxonomy(
            set(v[0] for v in get_ncbi_taxa().get_name_translator(selected_nodes).values())
        ).unique()
        # model_provided: dict[TaxName, list[ModelGeneticTool] | str] = {
        #     v["species"]: v["tools"] for v in (chatbot_provided or {}).get("results", [])
        # }
        model_provided = self._collapse_to_single_entry(chatbot_provided)
        for values in model_provided.values():
            for value in cast(list, values):
                value["references"] = (
                    ",".join(map(str, value["references"]))
                    if isinstance(value["references"], list)
                    else value["references"]
                )
        collected_tool_dfs = [
            (
                viral_ldf.match(bd_df["species-taxid"])
                .join(bd_df.lazy(), how="left", left_on=viral_ldf.host_id_col, right_on="species-taxid")
                .unique(viral_ldf.tool_name)
                .groupby("species")
                .agg(viral_ldf.tool_name, viral_ldf.tool_id_col)
            ).collect()
            for viral_ldf in self.genetic_tools
        ]

        return self._to_summary_rows(bd_df, collected_tool_dfs, model_provided)
