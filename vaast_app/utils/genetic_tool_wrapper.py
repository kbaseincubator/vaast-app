"""
Logic around wrapping genetic tool databases in generic implementations that are easier to search
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Iterable, cast

import polars as pl

_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


@dataclass
class GeneticToolDB:
    """
    Wrapper around genetic tool database
    """

    database_name: str
    host_id_col: str
    tool_id_col: str
    tool_name: str
    select_cols: Iterable[str | pl.Expr]
    file: Path
    _ldf: pl.LazyFrame | None = None

    def load_file_to_ldf(self) -> "GeneticToolDB":
        """
        Load parquet file into GeneticTool instance
        :return: ``GeneticTool`` instance
        """
        self._ldf = pl.scan_parquet(self.file).select(
            self.host_id_col, self.tool_id_col, self.tool_name, *self.select_cols
        )
        return self

    def match(self, taxids: Collection[int]) -> pl.LazyFrame:
        """
        Collect entries whose ``self.host_id_col`` is in provided ``taxids``
        :param taxids: Iterable of taxids to collect
        :return: Matching LazyFrame
        """
        return cast(pl.LazyFrame, self._ldf).filter(pl.col(self.host_id_col).is_in(taxids))


class GeneticToolDBList(list[GeneticToolDB]):
    """
    List of ``GeneticTool`` instances representing loose wrappers around parquet file databases
    """

    def __init__(self, searchers: Iterable[GeneticToolDB]):
        """
        Generate list of searchers

        :param searchers: GeneticTool instances
        """
        super().__init__([searcher.load_file_to_ldf() for searcher in searchers])

    def get(self, db_name: str) -> GeneticToolDB | None:
        """
        Get the tool database associated with provided ``db_name``

        :param db_name: Name of database
        :return: ``GeneticTool`` instance, or None if ``db_name`` is not present in ``self``
        """
        for tool_ldf in self:
            if tool_ldf.database_name == db_name:
                return tool_ldf
        return None


@lru_cache()
def load_genetic_tools() -> GeneticToolDBList:
    return GeneticToolDBList(
        [
            GeneticToolDB(
                database_name="Virus Host DB",
                host_id_col="host tax id",
                tool_id_col="virus tax id",
                tool_name="virus name",
                select_cols=["virus lineage"],
                file=Path("data/virus_host_db.parquet"),
            ),
            GeneticToolDB(
                database_name="Phage-Host DB",
                host_id_col="host_tax_id",
                tool_id_col="virus tax id",
                tool_name="virus name",
                select_cols=["lineage", "host_lineage"],
                file=Path("data/phd_db.parquet"),
            ),
            GeneticToolDB(
                database_name="PLSDB",
                host_id_col="host tax id",
                tool_id_col="plasmid id",
                tool_name="plasmid name",
                select_cols=[],
                file=Path("data/pls_db.parquet"),
            ),
        ]
    )


def get_unique_tool_count(tax_df: pl.DataFrame, genetic_tools: GeneticToolDBList) -> int:
    tools = set()
    for tool_db in genetic_tools:
        tool_df = tool_db.match(tax_df["species-taxid"]).select(pl.col(tool_db.tool_id_col).unique()).collect()
        tools.update(tool_df[tool_db.tool_id_col])
    return len(tools)


class _ToolMatcher:
    _TOOLS: pl.DataFrame | None = None

    @staticmethod
    def get(tax_df: pl.DataFrame, genetic_tools: GeneticToolDBList) -> pl.DataFrame:
        if _ToolMatcher._TOOLS is None:
            for tool_db in genetic_tools:
                tool_df = (
                    tool_db.match(tax_df["species-taxid"])
                    .group_by(tool_db.host_id_col)
                    .agg(pl.col(tool_db.tool_id_col).n_unique().alias(tool_db.database_name))
                )
                tax_df = (
                    tax_df.lazy()
                    .join(tool_df, how="left", left_on="species-taxid", right_on=tool_db.host_id_col)
                    .collect()
                )
            _ToolMatcher._TOOLS = (
                tax_df.with_columns(
                    (
                        pl.concat_list([tool_db.database_name for tool_db in genetic_tools])
                        .list.drop_nulls()
                        .list.len()
                        .alias("n_non_null")
                    )
                )
                .filter(pl.col("n_non_null").ge(1))
                .drop("n_non_null")
                .group_by("species-taxid")
                .agg(
                    pl.col(_RANKS).first(),
                    pl.col([tool_db.database_name for tool_db in genetic_tools]).fill_null(0).sum(),
                )
                .with_columns(pl.sum_horizontal([tool_db.database_name for tool_db in genetic_tools]).alias("tool-sum"))
            )
        return _ToolMatcher._TOOLS


def match_to_tools(tax_df: pl.DataFrame, genetic_tools: GeneticToolDBList) -> pl.DataFrame:
    return _ToolMatcher.get(tax_df, genetic_tools)
