import polars as pl
from dash import html, no_update
from dash.development.base_component import Component

from vaast_app.utils.bacdive_utils import load_bacdive_data
from vaast_app.utils.genetic_tool_wrapper import load_genetic_tools, match_to_tools

_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


def set_search_options(search_value: str) -> list[dict[str, str | Component]]:
    if not search_value or len(search_value) < 4:
        return no_update
    genetic_tools = load_genetic_tools()
    tax_df = load_bacdive_data()
    data_df = (
        match_to_tools(tax_df, genetic_tools)
        .lazy()
        .filter(pl.col("species").str.to_lowercase().str.contains(search_value.lower()))
        .select(pl.concat_str(_RANKS[1:], separator=";;").alias("tax-string"), "species")
        .sort("species")
        .collect()
    )
    return [
        {
            "label": html.Span(
                [
                    html.Div(species, className="pt-2"),
                    html.P(tax_string, className="text-body-secondary lead", style={"font-size": "10px"}),
                ],
                className="mt-2",
            ),
            "value": species,
        }
        for (tax_string, species) in data_df.iter_rows()
    ]
