"""Strain stats region"""

from typing import Iterable

import dash_bootstrap_components as dbc
import polars as pl
from dash import Input, Output, dash_table, dcc, html
from dash.development.base_component import Component
from ete3 import NCBITaxa

from vaast_app.render import Render
from vaast_app.utils.bacdive_utils import BacdiveSummary, BacdiveSummaryList
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.summary_region import SummaryRow, summary_row


class StrainStats(Render):
    """Strain stats region"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app.callback(
            Output("strain-stats-container", "children"),
            Output("strain-stats-loading", "children"),
            Input("stats-display-by", "value"),
            Input("results-tree", "selected_nodes"),
            prevent_initial_call=True,
        )(self._show_strain_stats)

    def _set_layout(self) -> Component:
        return dbc.Col(
            id="stats-col",
            children=[
                html.Span(
                    [
                        html.H4("Strain stats"),
                        html.Span(
                            [
                                "View by:",
                                dcc.RadioItems(
                                    ["Species", "Reference"],
                                    inline=True,
                                    id="stats-display-by",
                                    value="Species",
                                    className="ms-2",
                                    labelClassName="pe-1",
                                ),
                            ],
                            className="d-flex justify-content-end",
                        ),
                    ],
                    className="d-flex justify-content-between",
                ),
                dbc.Container(className="g-0 overflow-auto", id="strain-stats-container", style={"height": "32vh"}),
            ],
            className="border border-secondary border-bottom-0 pt-1 pb-2 col-6",
        )

    @staticmethod
    def _to_table(values: list[dict[str, str]]) -> dash_table.DataTable:
        return dash_table.DataTable(
            values,
            style_table={"overflowX": "scroll", "textAlign": "left"},
            page_size=5,
            style_header={"textAlign": "left"},
            page_action="native",
            columns=[{"name": col, "id": col, "presentation": "markdown"} for col in list(values[0].keys())],
            filter_action="native",
        )

    @staticmethod
    def _by_data_type_summary_row(strain_name: str, entries: Iterable[BacdiveSummary]) -> SummaryRow | None:
        out: list[list[dict]] = [[] for _ in range(6)]
        for entry in entries:
            for tax_info in sorted(entry.ncbi_tax, key=lambda v: v.ref)[-1:]:
                out[0].append(
                    {
                        "BD ref": entry.ref,
                        "Ref": tax_info.ref,
                        "Level": tax_info.level,
                        "NCBI Tax ID": tax_info.tax_id,
                        "Name": tax_info.name,
                        "Description": entry.description,
                    }
                )
            for media_info in sorted(entry.culture_mediums, key=lambda v: v.ref):
                out[1].append(
                    {
                        "BD ref": entry.ref,
                        "Ref": media_info.ref,
                        "Media": media_info.name,
                        "Composition": media_info.composition,
                    }
                )
            for growth_info in sorted(entry.culture_temps, key=lambda v: v.ref):
                out[2].append(
                    {
                        "BD ref": entry.ref,
                        "Ref": growth_info.ref,
                        "Type": growth_info.type,
                        "Temperatures": growth_info.temperature,
                        "Range": growth_info.range,
                    }
                )
            for enzyme_info in sorted(entry.enzymes, key=lambda v: v.ref):
                out[3].append(
                    {
                        "BD ref": entry.ref,
                        "Ref": enzyme_info.ref,
                        "EC": enzyme_info.ec_annotation,
                        "Name": enzyme_info.value,
                        "Activity": enzyme_info.activity,
                    }
                )
            for met_info in sorted(entry.metabolites, key=lambda v: v.ref):
                out[4].append(
                    {
                        "BD ref": entry.ref,
                        "Ref": met_info.ref,
                        "Name": met_info.metabolite,
                        "Chebi ID": met_info.chebi_id,
                        "Kind tested": met_info.utilization_kind,
                        "Utilization": met_info.utilization_activity,
                    }
                )
            for papers_info in sorted(entry.papers, key=lambda v: v.ref):
                for paper in papers_info.papers:
                    out[5].append(
                        {
                            "DOI": (
                                f"[https://doi.org/{doi}](https://doi.org/{doi})" if (doi := paper.doi) != "" else ""
                            ),
                            "BD ref": entry.ref,
                            "Ref": papers_info.ref,
                            "Title": paper.title,
                            "Journal": paper.journal,
                        }
                    )

        viz_out = []
        for name, data in zip(
            [
                ("Summary", "summary"),
                ("Isolation Media", "iso-med"),
                ("Growth Conditions", "grwth-stats"),
                ("Enzymatic Activity", "enz-act"),
                ("Metabolite Utilization", "met-util"),
                ("References", "refs"),
            ],
            out,
            strict=True,
        ):
            if len(data) == 0:
                continue
            viz_out.append(dbc.AccordionItem(StrainStats._to_table(data), title=name[0], item_id=name[1]))
        if len(viz_out) == 0:
            return None
        return summary_row(strain_name, iter(viz_out))

    def _display_by_data_type(self, bd_tax_df: pl.DataFrame, ncbi_client: NCBITaxa) -> list[SummaryRow]:
        out = []
        for row in bd_tax_df.iter_rows(named=True):
            bd_ids: Iterable[str] = row["BacDive ID"]
            entries = self.bacdive_searcher.get(bd_ids, ncbi_client)
            species, species_replaced = row["species"][0], row["species-replaced"]
            name = f"{species_replaced} ({species})" if species != species_replaced else species_replaced
            if (s_row := StrainStats._by_data_type_summary_row(name, entries)) is not None:
                out.append(s_row)
        return out

    def _display_by_reference(self, bd_tax_df: pl.DataFrame, ncbi_client: NCBITaxa) -> list[SummaryRow]:
        out = []
        data_types = ["culture_mediums", "metabolites", "enzymes"]
        for row in bd_tax_df.iter_rows(named=True):
            bd_ids: Iterable[str] = row["BacDive ID"]
            entries: BacdiveSummaryList = self.bacdive_searcher.get(bd_ids, ncbi_client)
            for bd_ref, entry_set in entries.get_entry_refs(*data_types).items():
                species, species_replaced = row["species"][0], row["species-replaced"]
                name = f"{species_replaced} ({species})" if species != species_replaced else species_replaced
                if (s_row := StrainStats._by_data_type_summary_row(f"{str(bd_ref)}: {name}", entry_set)) is not None:
                    out.append(s_row)
        return out

    @Render.with_update(1, "after")
    def _show_strain_stats(self, display_by: str, selected_nodes: list[str]) -> list[SummaryRow]:
        ncbi_client = get_ncbi_taxa()
        translator = ncbi_client.get_name_translator(selected_nodes)
        tax_df = self.bacdive_searcher.get_taxonomy(set(v[0] for v in translator.values()))
        # Map to current names in NCBI
        bd_tax_df = (
            tax_df.with_columns(
                (
                    pl.struct("species-taxid", "species")
                    .replace(ncbi_client.get_taxid_translator(tax_df["species-taxid"]), default=pl.col("species"))
                    .alias("species-replaced")
                )
            )
            .groupby("species-replaced")
            .agg(pl.col("ID").alias("BacDive ID"), "species")
            .sort("species-replaced")
        )
        if display_by == "Species":
            return self._display_by_data_type(bd_tax_df, ncbi_client)
        if display_by == "Reference":
            return self._display_by_reference(bd_tax_df, ncbi_client)
        raise NotImplementedError(f"radio selection '{display_by}' is not yet implemented")
