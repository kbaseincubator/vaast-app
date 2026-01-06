from collections.abc import Callable
from typing import cast

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, get_asset_url, html, no_update, register_page
from dash.development.base_component import Component

from vaast_app.utils.bacdive_utils import load_bacdive_data
from vaast_app.utils.genetic_tool_wrapper import get_unique_tool_count, load_genetic_tools, match_to_tools
from vaast_app.utils.search_by_taxa import set_search_options

register_page(__name__, path="/")

_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


genetic_tools = load_genetic_tools()
tax_df = load_bacdive_data()
matched_df = match_to_tools(tax_df, genetic_tools)


def treemap() -> go.Figure:
    colors_by_depth = [
        "lightgray",
        "#006e5b",
        "#009688",
        "#66c0b8",
        "#a8cdea",
        "#c7dbee",
        "#dee9f5",
    ]

    fig = px.treemap(matched_df.to_pandas(), path=_RANKS, custom_data=["Virus Host DB", "Phage-Host DB", "PLSDB"])
    cast(dict, fig.data[0])["marker"]["colors"] = [
        colors_by_depth[len(d.split("/")) - 1] for d in cast(dict, fig.data[0])["ids"]
    ]
    fig = fig.update_layout(margin=dict(l=20, r=20, t=20, b=20)).update_traces(
        marker=dict(cornerradius=5),
        hovertemplate="<b>%{label}</b><br><br>%{id}<br><br><b>VHD:</b>&nbsp;%{customdata[0]}<br><b>PHD:</b>&nbsp;%{customdata[1]}<br><b>PLSDB:</b>&nbsp;%{customdata[2]}<extra></extra>",
    )
    # TODO: Write total tool count in upper-level hierarchies
    fig.data[0].customdata = [
        v if fig.data[0].labels[i] in matched_df["species"] else ["", "", ""]
        for i, v in enumerate(fig.data[0].customdata)
    ]
    return fig


text = [
    [
        html.P(
            "Novel synthetic biology workflows and experiments require large literature searches in preparation to an experiment"
        ),
        html.P(
            (
                "With the size of the literature corpus growing, modern "
                "literature reviews would require 100's to 1000's of searches to adequately cover the breadth "
                "of available manuscripts."
            )
        ),
        html.P(
            (
                "VAAST links existing data in various databases with a powerful LLM trained to assist in genetic engineering tasks."
            )
        ),
        html.P(
            (
                "VAAST provides a set of curated manuscripts and preprints to guide researchers towards their project goals."
            )
        ),
        html.Img(src=get_asset_url("workflow.jpg"), width="750px", className="d-lg-block d-none"),
        # html.Img(
        #     src=get_asset_url("workflow-edits-for-small-screens.png"), height="300px", className="d-block d-lg-none"
        # ),
    ],
    [
        html.P("We compiled 24,856 publications to train an LLM for synthetic biology applications", className="lead"),
        html.P(
            "These publications were selected by genetic engineering keywords, and then validated by LLM prompting for relevancy"
        ),
        html.P(
            (
                html.Span("14,359 PMC publications", className="lead"),
                html.Img(
                    src="https://pmc.ncbi.nlm.nih.gov/static/img/pmc-logo.svg",
                    width="450px",
                    className="d-lg-block d-none",
                ),
            )
        ),
        html.P(
            (
                html.Span("10,503 bioRxiv publications", className="lead"),
                html.Img(
                    src="https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png",
                    width="450px",
                    className="d-lg-block d-none",
                ),
            )
        ),
    ],
    [
        html.P(
            "Built using the BacDive database, VAAST provides an isolation-driven knowledgebase, "
            "incorporating the full NCBI database for exploration into unisolated organisms."
        ),
        html.P(
            (
                "Isolation information including ...",
                "Taxonomy information updated frequently ",
                "Supports older nomenclatures",
            )
        ),
        html.P(
            html.Img(src="https://bacdive.dsmz.de/images/logo/BacDiveLogo.svg", width="450px"),
        ),
    ],
    [
        html.P("Phage and plasmid databases display alongside LLM results to highlight genetic engineerability."),
        html.P(
            [
                html.Span("Virus Host Database", className="h5 pe-2"),
                html.Span(
                    "NCBI/RefSeq and GenBank viruses and their hosts (plus UniProt and ViralZone)",
                    className="lead",
                ),
            ]
        ),
        html.P(
            [
                html.Span("Phage & Host Daily", className="h5 pe-2"),
                html.Span("Catalog of viral/prokaryotic interations", className="lead"),
            ]
        ),
        html.P(
            [
                html.Span("PLSDB", className="h5 pe-2"),
                html.Span("Bacterial plasmids from NCBI nucleotide database", className="lead"),
            ]
        ),
    ],
]


def _names_join(fxn: Callable[..., str]) -> Callable[..., str]:
    def f(*class_name: str) -> str:
        names = " " + " ".join(class_name)
        if len(names) == 1:
            names = ""
        return fxn() + names

    return f


@_names_join
def row() -> str:
    return "d-flex"


@_names_join
def offset_mid() -> str:
    return "col-10 offset-1"


@_names_join
def container_fluid() -> str:
    return "container-fluid g-0"


def title_row() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("VAAST", className="display-1"),
                            html.Small("by KBase", className="text-body-secondary"),
                        ],
                        className=row("ps-3 align-items-end"),
                    ),
                    html.Div(
                        [
                            html.H5("Genetic engineering toolkit powered by AI", className="text-body-secondary"),
                        ],
                        className=offset_mid("ps-2 pe-2 pt-1 pb-4"),
                    ),
                    html.Div(
                        [
                            html.Div("Search:", className="h5"),
                            dcc.Dropdown(
                                options=[],
                                placeholder="Enter species...",
                                className="container-fluid rounded multi-line-dropdown",
                                id="dropdown-user-search",
                                style={"whiteSpace": "normal", "lineHeight": "15px"},
                            ),
                        ],
                        className=offset_mid("ps-2 pe-2 pt-1 pb-4"),
                    ),
                ],
                className="border rounded border-secondary mh-25 title-row-bg w-75",
            ),
        ],
        className=row("h-50 img-header-bg align-items-center justify-content-center"),
    )


def tree_summary_row() -> html.Div:
    return html.Div(
        [
            html.Img(
                src=get_asset_url("IMG_0035.jpeg"),
                className="tree-img",
            ),
        ],
        className=row("align-items-center justify-content-center mt-4 mb-4"),
    )


def site_summary_row() -> html.Div:
    # TODO: Transpose rows to columns to reduce whitespace on larger screens
    return html.Div(
        [
            html.Div(
                [
                    # TODO: align-items-center is insufficient for layout on small screens - should align by bolded text label, not by whole region
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Img(
                                        src=get_asset_url("virus-infection-istock.jpg"),
                                        style={"border-radius": "50%"},
                                        className="site-summary-img img-fluid",
                                    ),
                                    html.P(
                                        [
                                            html.P(
                                                f"{get_unique_tool_count(tax_df, genetic_tools):,} genetic tools",
                                                className="h5",
                                            ),
                                            html.P("collected from VHD, PHD, and PLSDB", className="lead text-center"),
                                        ],
                                        className=row("flex-column align-items-center mt-1"),
                                    ),
                                ],
                                className=row(
                                    "col-12 col-md-4 flex-column justify-content-center align-items-center me-lg-2"
                                ),
                            ),
                            html.Div(
                                [
                                    html.Img(
                                        src=get_asset_url("bacteria-istock.jpg"),
                                        style={"border-radius": "50%"},
                                        className="site-summary-img img-fluid",
                                    ),
                                    html.P(
                                        [
                                            html.P(f"{len(matched_df):,} species", className="h5"),
                                            html.P("collected from BacDive database", className="lead text-center"),
                                        ],
                                        className=row("flex-column align-items-center mt-1"),
                                    ),
                                ],
                                className=row(
                                    "col-12 col-md-4 flex-column justify-content-center align-items-center me-lg-2"
                                ),
                            ),
                            html.Div(
                                [
                                    html.Img(
                                        src=get_asset_url("literature-istock.jpg"),
                                        style={"border-radius": "50%"},
                                        className="site-summary-img img-fluid",
                                    ),
                                    html.P(
                                        [
                                            html.P(">24K articles", className="h5"),
                                            html.P(
                                                "encompassing BioRxiv and PMC",
                                                className="lead text-center",
                                            ),
                                        ],
                                        className=row("flex-column align-items-center mt-1"),
                                    ),
                                ],
                                className=row(
                                    "col-12 col-md-4 flex-column justify-content-center align-items-center me-lg-2"
                                ),
                            ),
                        ],
                        className=row("flex-column flex-md-row"),
                    ),
                ],
                className=offset_mid(),
            )
        ],
        className=row("pt-4 bg-light h-md-50 align-items-center"),
    )


def about_site_row() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("What is VAAST?", className="h2"),
                                    html.P("Visual AI Assistant SynBio Toolkits", className="lead"),
                                    html.Div(
                                        id="about-text-display",
                                        style={"height": "45vh"},
                                        children=text[0],
                                        className="g-0 justify-content-center justify-content-lg-left d-flex flex-column flex-lg-row d-lg-block",
                                    ),
                                ],
                                className="col-12 col-lg-7",
                            ),
                            html.Div(
                                [
                                    html.Img(
                                        src=get_asset_url("wide-image.jpg"),
                                        style={
                                            "border-top-left-radius": "50%",
                                            "border-bottom-left-radius": "50%",
                                        },
                                        className="img-fade",
                                    )
                                ],
                                className="col-lg-5 d-flex justify-content-end d-none d-lg-block",
                            ),
                        ],
                        className=row("justify-content-start align-items-center ps-2 pe-2 pt-1 pb-1"),
                    ),
                    html.Div(
                        dbc.Pagination(
                            id="about-text-page",
                            active_page=1,
                            min_value=1,
                            max_value=len(text),
                            size="md",
                            class_name="btn-secondary",
                        ),
                        className=row("ps-2 pe-2 pt-1 pb-1 justify-content-center"),
                    ),
                ],
                className=offset_mid(),
            )
        ],
        className=row("mh-50 mt-4"),
    )


def collection_overview_row() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Collection overview", className="h3"),
                                ]
                            ),
                            html.Div([dcc.Graph(figure=treemap(), style={"height": "65vh"}, id="treemap-collection")]),
                        ],
                        className="justify-content-start align-items-center ps-2 pe-2",
                    ),
                ],
                className=offset_mid(),
            )
        ],
        className=row("h-75 bg-light mt-4 align-items-center"),
    )


def redirect_to_visualizer() -> Component:
    return dcc.Link(["Visualizer"], className="btn btn-secondary floating-btn", href="/visualizer")


layout = html.Div(
    [
        # App outermost container
        html.Div(
            [
                html.Div(
                    [
                        dbc.NavItem(
                            html.A(
                                html.Img(
                                    src="https://www.kbase.us/wp-content/uploads/sites/6/2020/08/k-base_logo.svg",
                                    style={"height": "75px"},
                                    # className="h-75 ms-3",
                                    className="w-75",
                                ),
                                href="/",
                            ),
                            className="text-center",
                        ),
                    ],
                    className=row(
                        "w-100 mb-4 nav-hov position-absolute top-0 start-0 align-items-center justify-content-between flex-column flex-sm-row mt-3"
                    ),
                    style={"height": "95px"},
                ),
                title_row(),
                # tree_summary_row(),
                about_site_row(),
                site_summary_row(),
                collection_overview_row(),
                redirect_to_visualizer(),
                # html.Footer([html.Div("Footer Placeholder", className="h3")], className="ms-2"),
            ],
            className="vh-100",
        ),
    ],
)


@callback(
    Output("about-text-display", "children"),
    Input("about-text-page", "active_page"),
    prevent_initial_call=True,
)
def set_about_text(n: int) -> Component:
    if not n:
        return text[0]
    return text[n - 1]


callback(
    Output("dropdown-user-search", "options"),
    Input("dropdown-user-search", "search_value"),
    prevent_initial_call=True,
)(set_search_options)


@callback(
    Output("store-main-query", "data", allow_duplicate=True),
    Input("dropdown-user-search", "value"),
    prevent_initial_call=True,
)
def set_landing_query(v) -> dict[str, list[str]]:
    if v is not None:
        return {"query": [v]}
    return no_update


@callback(
    Output("url", "href", allow_duplicate=True),
    Input("dropdown-user-search", "value"),
    prevent_initial_call=True,
)
def visualizer_with_value(value: str | None) -> str:
    if value is not None:
        return "/visualizer"
    return no_update


@callback(
    Output("store-main-query", "data"),
    Output("url", "href", allow_duplicate=True),
    Input("treemap-collection", "clickData"),
    prevent_initial_call=True,
)
def visualize_from_treemap(value) -> tuple[dict[str, list[str]], str]:
    if value is None:
        return no_update, no_update
    if value["points"][0]["customdata"] == ["", "", ""]:
        return no_update, no_update
    return {"query": [value["points"][0]["label"]]}, "/visualizer"


if __name__ == "__main__":
    ...
