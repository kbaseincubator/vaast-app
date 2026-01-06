"""
Utility classes and functions for interacting with the ete3 NCBITaxa and PhyloNode classes.
"""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import NewType, TypedDict, cast

import polars as pl
from ete3 import NCBITaxa, PhyloNode

from vaast_app.utils.bacdive_utils import BacdiveAPISearcher, BacdiveMetadata
from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.type_utils import TaxID, TaxName


class TreeJSON(TypedDict):
    """
    JSON data for visualizing tree
    """

    newick: str
    metadata: BacdiveMetadata
    selected_nodes: list[str]


class TreeResultData(TypedDict):
    """
    Results from querying NCBI tree
    """

    failed: list[TaxName]
    tree: TreeJSON


NonSpeciesEntry = NewType("NonSpeciesEntry", str)


class TreeUtils:
    """
    Search NCBI taxonomy tree for various taxonomic names. Identify entries which contain BacDive entries.
    """

    def __init__(self, bacdive_searcher: BacdiveAPISearcher, names_parquet: Path | None = None):
        self._bacdive_searcher = bacdive_searcher
        self._names_pkt: pl.LazyFrame | None = None
        if names_parquet is not None:
            self._names_pkt = pl.scan_parquet(names_parquet)

    @dataclass(frozen=True)
    class SearchResult:
        """
        Packaged query results summarizing the failed query ids and the resulting JSON tree data to provide to the
        PhylotreeDash custom component.
        """

        _tree: PhyloNode | None
        _failed_queries: list[TaxName]
        _bacdive_searcher: BacdiveAPISearcher
        _selected_nodes: list[str]

        def is_valid(self) -> bool:
            """
            Results contain a valid PhyloNode head. This occurs when at least 1 query term returns results.

            :return: True/False
            """
            return self._tree is not None

        @property
        def tree(self) -> PhyloNode:
            """
            Pointer to tree.

            :return: Root node to NCBI taxa tree that contains results for terms in user query results
            :raises ValueError: if tree is invalid
            """
            if not self.is_valid():
                raise ValueError()
            return cast(PhyloNode, self._tree)

        def to_data(self, chatbot_provided: list[ChatbotPayload] | None = None) -> TreeResultData:
            """
            Convert ``SearchResult`` class to a dictionary in the form:

            {"failed": [], "tree": {"newick": "", "metadata: [], "selected_nodes": []}}

            :return: Dictionary of summarized data which can convert to valid JSON
            """
            data: TreeResultData = {"failed": self._failed_queries, "tree": self._to_tree_json(chatbot_provided)}
            return data

        def _to_tree_json(self, chatbot_provided: list[ChatbotPayload] | None) -> TreeJSON:
            """
            Convert tree to expected format for PhylotreeDash component.

            This contains a newick string generated with ete3 (format=3, quoted_node_names=True), a list of nodes
            that contain BacDive annotations, and an empty list of selected nodes (for clearing the selected nodes in
            the tree).

            :return: Dictionary of search results
            """
            if not self.is_valid():
                return {"newick": "", "metadata": {}, "selected_nodes": []}
            tree = cast(PhyloNode, self._tree).copy()
            translator = get_ncbi_taxa().get_taxid_translator([node.name for node in cast(Generator, tree.traverse())])
            for node in cast(Generator, tree.traverse()):
                node.name = translator[int(node.name)]
                # Quote characters cause issues for newick format
                node.sci_name = node.sci_name.replace("'", "")
            return {
                "newick": cast(
                    str, tree.write(format=3, quoted_node_names=True, features=["sci_name", "taxid", "rank"])
                ),
                "metadata": self._bacdive_searcher.metadata(cast(PhyloNode, self._tree), translator, chatbot_provided),
                "selected_nodes": self._selected_nodes,
            }

    def _search_names_parquet(
        self, names: list[TaxName], ncbi_client: NCBITaxa
    ) -> dict[str, tuple[list[TaxID], list[str]]]:
        if len(names) == 0 or self._names_pkt is None:
            return {}
        # pylint: disable=fixme
        # TODO: 'Pseudomonas stutzeri' also collects 'Pseudomonas balearica' somehow.
        out: dict[str, tuple[list[TaxID], list[str]]] = {}
        for name in names:
            # Collect all taxids with entries that start with the query name
            entry_data = self._names_pkt.filter(pl.col("name_txt").str.contains(name)).select("tax_id").collect()
            # Get taxid and names of entries that are in `entry_data` and that are "scientific name" entry types
            sci_names = (
                self._names_pkt.filter(
                    (pl.col("tax_id").is_in(entry_data["tax_id"]) & pl.col("name class").eq("scientific name"))
                ).select("tax_id", "name_txt")
            ).collect()
            potential_entries = (sci_names["tax_id"].unique().to_list(), sci_names["name_txt"].unique().to_list())
            value = out[name] = ([], [])
            for taxid, name_txt in zip(*potential_entries, strict=True):
                try:
                    if 2 not in (ncbi_client.get_lineage(taxid) or []):
                        continue
                except ValueError:
                    continue
                value[0].append(taxid)
                value[1].append(name_txt)
        return out

    def query_for_tax_names(self, tax_names: list[str]) -> tuple[set[TaxID], set[NonSpeciesEntry], list[TaxName]]:
        """
        Query NCBI taxonomy database for a list of arbitrary tax names

        `tax_names` will be modified in place

        :param tax_names: List of names to search
        :return: Tuple of (found tax ids, non species entries, failed entries)
        """
        ncbi_client = get_ncbi_taxa()
        translator = ncbi_client.get_name_translator(tax_names)
        query_taxids = set()
        failed = []
        non_species_entries = set()
        valid_ranks = ("species", "subspecies", "strain")
        for query_tax in tax_names.copy():
            taxids = translator.get(query_tax, None)
            # Not found, or described uncharacterized sequences
            if not taxids or 32644 in taxids:
                failed.append(TaxName(query_tax))
                continue
            # Is a root-level identifier or not Bacteria
            if taxids[0] < 5 or 2 not in (ncbi_client.get_lineage(taxids[0]) or []):
                continue
            # Confirm entry is species-level
            rank = ncbi_client.get_rank([taxids[0]]).get(taxids[0], "")
            if rank in valid_ranks:
                query_taxids.add(taxids[0])
                tax_names.append(query_tax)
            # Collect valid non-species entries
            elif rank != "":
                non_species_entries.add(taxids[0])
            # Entry not found
            else:
                failed.append(TaxName(query_tax))
        # Search for failed terms in complete names dump
        try_2_failed = []
        for failed_name, (deep_result_search, found_names) in self._search_names_parquet(failed, ncbi_client).items():
            if len(deep_result_search) == 0:
                try_2_failed.append(TaxName(failed_name))
            else:
                query_taxids.update(deep_result_search)
                tax_names.remove(failed_name)
                tax_names.extend(found_names)
        return query_taxids, non_species_entries, try_2_failed

    def bacdive_based_search(self, query_tax_names: list[str]) -> "TreeUtils.SearchResult":
        """
        Search for entries that are described by the `BacdiveAPISearcher` utility class

        :param query_tax_names: List of taxa to search
        :return: Results of search
        """
        # Reinsert operation - query for name->taxid, then reinsert taxid->name to handle updates in NCBI databases
        # TODO: Optimize this
        ncbi_client = get_ncbi_taxa()
        query_taxids = []
        for taxids in ncbi_client.get_name_translator(query_tax_names).values():
            query_taxids.extend(taxids)
        query_tax_names.extend(ncbi_client.get_taxid_translator(query_taxids).values())
        query_tax_names = list(set(query_tax_names))
        query_taxids, non_species_entries, failed = self.query_for_tax_names(query_tax_names)
        query_tax_names = list(set(query_tax_names))
        # Populate species entries by finding their genus and getting all BacDive entries with the same genus
        query_tax_df = self._bacdive_searcher.get_taxonomy(query_taxids)
        for genus_tax in query_tax_df.select(*self._bacdive_searcher.ranks[:-1]).unique().iter_rows():
            query_taxids.update(
                self._bacdive_searcher.filter_by_ranks(genus_tax)
                .select("species-taxid")
                .drop_nulls()
                .unique()
                .collect()["species-taxid"]
            )
        # Incorporate BacDive entries that match at non-genus level and include
        for non_species in non_species_entries:
            rank = ncbi_client.get_rank([non_species])[non_species]
            rank_idx = self._bacdive_searcher.ranks.index(rank) if rank in self._bacdive_searcher.ranks else None
            if rank_idx is None:
                continue
            non_species_tx = ncbi_client.get_taxid_translator([non_species])[non_species]
            entries = self._bacdive_searcher.summary_df.filter(pl.col(rank).eq(non_species_tx)).drop_nulls(
                ["species-taxid"]
            )
            query_taxids.update(entries.collect()["species-taxid"])
        if len(query_taxids) == 0:
            return TreeUtils.SearchResult(None, failed, self._bacdive_searcher, query_tax_names)
        tree = ncbi_client.get_topology(list(query_taxids)[:500], intermediate_nodes=True, collapse_subspecies=True)
        return TreeUtils.SearchResult(tree, failed, self._bacdive_searcher, query_tax_names)
