"""Helper function for summary region"""
from typing import NewType, Iterator

import dash_bootstrap_components as dbc

SummaryRow = NewType("SummaryRow", dbc.Row)  # type: ignore[valid-newtype]


def summary_row(strain_name: str, regions: Iterator[dbc.AccordionItem]) -> SummaryRow:
    """Generate bootstrap `Row` containing a card/card-body item"""
    return SummaryRow(dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader(strain_name),
        dbc.CardBody(dbc.Accordion(list(regions), active_item=None, always_open=True))
    ]), className="mb-2 g-0"), className="g-0"))
