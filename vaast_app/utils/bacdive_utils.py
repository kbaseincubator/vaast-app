"""
Logic for BacDive-specific parsing and data manipulation.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

import numpy as np
import polars as pl
from ete3 import NCBITaxa, PhyloNode

from vaast_app.utils.chatbot.chatbot_utils import ChatbotPayload
from vaast_app.utils.genetic_tool_wrapper import GeneticToolDBList, load_genetic_tools
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.read_utils import json_unzip
from vaast_app.utils.type_utils import (
    BacdiveEntryRefID,
    BacdiveID,
    BacdiveRef,
    Feature,
    HeatmapID,
    TaxID,
    TaxName,
)


def _trim(value: str) -> str:
    """
    Replace newline character in string with a single space

    :param value: Value (potentially containing newline)
    :return: Value with newline replaced
    """
    return value.replace("\n", " ")


@dataclass
class MetaboliteUtilization:
    """
    Parsed "metabolite utilization" section of BacDive API search results.
    """

    ref: BacdiveEntryRefID
    metabolite: str
    utilization_activity: bool
    utilization_kind: str
    chebi_id: int

    @staticmethod
    def parse_metabolites(
        metabolites: dict[str, int | str] | list[dict[str, int | str]],
    ) -> list["MetaboliteUtilization"]:
        """
        Convert JSON data to class

        :param metabolites: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(metabolites, dict):
            return [
                MetaboliteUtilization(
                    metabolite=metabolites.get("metabolite", ""),
                    utilization_kind=metabolites.get("kind of utilization tested", ""),
                    utilization_activity=metabolites.get("utilization activity", "-") == "+",
                    ref=BacdiveEntryRefID(metabolites.get("@ref", -1)),
                    chebi_id=metabolites.get("Chebi-ID", -1),
                )
            ]
        out = []
        for metabolite_info in metabolites:
            out.append(
                MetaboliteUtilization(
                    metabolite=metabolite_info.get("metabolite", ""),
                    utilization_kind=metabolite_info.get("kind of utilization tested", ""),
                    utilization_activity=metabolite_info.get("utilization activity", "-") == "+",
                    ref=BacdiveEntryRefID(metabolite_info.get("@ref", -1)),
                    chebi_id=metabolite_info.get("Chebi-ID", -1),
                )
            )
        return out


@dataclass
class Paper:
    """
    Paper attached to a given BacdiveEntryRefID
    """

    title: str
    authors: str
    doi: str
    journal: str

    @staticmethod
    def parse_papers(data: dict[str, str] | list[dict[str, str]]) -> list["Paper"]:
        """
        Convert JSON data to class

        :param data: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(data, dict):
            return [
                Paper(
                    title=data.get("title", ""),
                    authors=data.get("authors", ""),
                    doi=data.get("doi", ""),
                    journal=data.get("journal", ""),
                )
            ]
        if isinstance(data, list):
            return [
                Paper(
                    title=paper.get("title", ""),
                    authors=paper.get("authors", ""),
                    doi=paper.get("DOI", ""),
                    journal=paper.get("journal", ""),
                )
                for paper in data
            ]
        raise RuntimeError()


@dataclass
class PaperReference:
    """
    Parsed literature references
    """

    ref: BacdiveEntryRefID
    papers: list[Paper]
    _JSON = dict[str, int | str | list[dict[str, str]]]

    @staticmethod
    def parse_paper_references(data: "PaperReference._JSON | list[PaperReference._JSON]") -> list["PaperReference"]:
        """
        Convert JSON data to class

        :param data: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(data, dict):
            return [
                PaperReference(
                    ref=BacdiveEntryRefID(data.get("@ref", -1)), papers=Paper.parse_papers(data.get("literature", []))
                )
            ]
        if isinstance(data, list):
            return [
                PaperReference(
                    ref=BacdiveEntryRefID(paper.get("@ref", -1)), papers=Paper.parse_papers(paper.get("literature", []))
                )
                for paper in data
            ]
        raise RuntimeError()


@dataclass
class EnzymeActivity:
    """
    Parsed "enzymes" section of BacDive API search results.
    """

    ref: BacdiveEntryRefID
    activity: bool
    value: str
    ec_annotation: str

    @staticmethod
    def parse_enzymes(enzymes: dict[str, int | str] | list[dict[str, int | str]]) -> list["EnzymeActivity"]:
        """
        Convert JSON data to class

        :param enzymes: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(enzymes, dict):
            return [
                EnzymeActivity(
                    activity=enzymes.get("activity", "-") == "+",
                    value=enzymes.get("value", ""),
                    ec_annotation=enzymes.get("ec", ""),
                    ref=BacdiveEntryRefID(enzymes.get("@ref", -1)),
                )
            ]
        out = []
        for enzyme_info in enzymes:
            out.append(
                EnzymeActivity(
                    activity=enzyme_info.get("activity", "-") == "+",
                    value=enzyme_info.get("value", ""),
                    ec_annotation=enzyme_info.get("ec", ""),
                    ref=BacdiveEntryRefID(enzyme_info.get("@ref", -1)),
                )
            )
        return out


@dataclass
class CultureTemperature:
    """
    Parse "culture temp" section of BacDive API search results.
    """

    ref: BacdiveEntryRefID
    growth: str
    type: str
    temperature: str
    range: str

    @staticmethod
    def parse_culture_temps(
        culture_temps: dict[str, int | str] | list[dict[str, int | str]],
    ) -> list["CultureTemperature"]:
        """
        Convert JSON data to class

        :param culture_temps: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(culture_temps, dict):
            return [
                CultureTemperature(
                    ref=BacdiveEntryRefID(culture_temps.get("@ref", -1)),
                    growth=culture_temps.get("growth", "unknown"),
                    type=culture_temps.get("type", ""),
                    temperature=culture_temps.get("temperature", ""),
                    range=culture_temps.get("range", ""),
                )
            ]
        return [
            CultureTemperature(
                ref=BacdiveEntryRefID(culture_temp.get("@ref", -1)),
                growth=culture_temp.get("growth", "unknown"),
                type=culture_temp.get("type", ""),
                temperature=culture_temp.get("temperature", ""),
                range=culture_temp.get("range", ""),
            )
            for culture_temp in culture_temps
        ]


@dataclass
class CultureMedium:
    """
    Parsed "culture medium" section of BacDive API search results.
    """

    ref: BacdiveEntryRefID
    name: str
    growth: bool
    composition: str

    @staticmethod
    def parse_culture_media(culture_media: dict[str, int | str] | list[dict[str, int | str]]) -> list["CultureMedium"]:
        """
        Convert JSON data to class

        :param culture_media: JSON data from BacDive ID query
        :return: Parsed JSON data
        """
        if isinstance(culture_media, dict):
            return [
                CultureMedium(
                    growth=culture_media.get("growth", "") == "yes",
                    name=_trim(culture_media.get("name", "")),
                    composition=culture_media.get("composition", ""),
                    ref=BacdiveEntryRefID(culture_media.get("@ref", -1)),
                )
            ]
        out = []
        for _culture_medium in culture_media:
            if isinstance(_culture_medium, dict):
                out.append(
                    CultureMedium(
                        growth=_culture_medium.get("growth", "") == "yes",
                        name=_trim(_culture_medium.get("name", "")),
                        composition=_culture_medium.get("composition", ""),
                        ref=BacdiveEntryRefID(_culture_medium.get("@ref", -1)),
                    )
                )
            elif isinstance(_culture_medium, list):
                for _culture_m in _culture_medium:
                    out.append(
                        CultureMedium(
                            growth=_culture_m.get("growth", "") == "yes",
                            name=_trim(_culture_m.get("name", "")),
                            composition=_culture_m.get("composition", ""),
                            ref=BacdiveEntryRefID(_culture_m.get("@ref", -1)),
                        )
                    )
            else:
                raise RuntimeError()
        return out


PRIORITY = ["species", "strain"]


@dataclass
class NCBITax:
    """
    Parsed "NCBI tax id" section of BacDive API search results.
    """

    ref: BacdiveEntryRefID
    name: TaxName
    tax_id: TaxID
    level: str

    @staticmethod
    def parse_tax_list(
        tax_info: dict[str, int | str] | list[dict[str, int | str]], ref: BacdiveEntryRefID, ncbi_client: NCBITaxa
    ) -> list["NCBITax"]:
        """
        Convert JSON data to class

        :param tax_info: JSON data from BacDive ID query
        :param ref: Assigned BacdiveEntryRefID
        :param ncbi_client: NCBITaxa instance
        :return: Parsed JSON data
        """
        not_found = [NCBITax(name=TaxName(""), tax_id=TaxID(-1), level="", ref=ref)]
        if isinstance(tax_info, dict):
            tax_id: int = tax_info.get("NCBI tax id", -1)
            if tax_id == -1:
                return not_found
            return [
                NCBITax(
                    name=TaxName(ncbi_client.get_taxid_translator([tax_id])[tax_id]),
                    tax_id=TaxID(tax_id),
                    level=tax_info["Matching level"],
                    ref=ref,
                )
            ]
        if isinstance(tax_info, list):
            out = []
            for priority in PRIORITY:
                for sci_name in tax_info:
                    if sci_name["Matching level"] == priority:
                        tax_id: int = sci_name.get("NCBI tax id", -1)
                        if tax_id == -1:
                            continue
                        out.append(
                            NCBITax(
                                name=TaxName(ncbi_client.get_taxid_translator([tax_id])[tax_id]),
                                tax_id=TaxID(tax_id),
                                level=priority,
                                ref=ref,
                            )
                        )
            if len(out) == 0:
                return not_found
            return out
        raise RuntimeError(tax_info)


@dataclass(eq=True)
class BacdiveSummary:
    """
    Summary of BacDive API query with data parsed for concomitant object attributes.
    """

    ref: BacdiveRef
    description: str
    ncbi_tax: list[NCBITax]
    culture_mediums: list[CultureMedium]
    culture_temps: list[CultureTemperature]
    metabolites: list[MetaboliteUtilization]
    enzymes: list[EnzymeActivity]
    papers: list[PaperReference]

    def __hash__(self) -> int:
        return hash(self.ref)

    def prune(self, entry_ref: BacdiveEntryRefID) -> "BacdiveSummary":
        """
        Collect summary entries with matching `BacdiveEntryRefID` values

        :param entry_ref: BD Entry reference ID
        :return: Summary object containing relevant data for `entry_ref`
        """
        out = BacdiveSummary(self.ref, "", [], [], [], [], [], [])
        for field in fields(self):
            if field.name in ("ref", "description"):
                continue
            new_attr = getattr(out, field.name)
            for old_attr in getattr(self, field.name):
                if old_attr.ref == entry_ref:
                    new_attr.append(old_attr)
        return out


class BacdiveSummaryList(Iterable[BacdiveSummary]):
    """
    Helper class to operations on a list of BacdiveSummary objects
    """

    def __init__(self, values: Iterable[BacdiveSummary]):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def get_entry_refs(self, *attrs: str) -> dict[BacdiveEntryRefID, set[BacdiveSummary]]:
        """
        Get map of @ref(BacdiveEntryRefID): [BacdiveSummary]

        :param attrs: List of attributes to collect
        :raises AttributeError: If attr is not a valid BacdiveSummary attribute
        :return: Mapping between entry references and their respective summaries
        """
        refs_map = defaultdict(set)
        for entry in self._values:
            for attr in attrs:
                bd_summary_attr = getattr(entry, attr)
                for value in bd_summary_attr:
                    if (ref := getattr(value, "ref", -1)) == -1:
                        continue
                    refs_map[ref].add(entry.prune(ref))
        return refs_map


BacdiveTreeResult = tuple[list[HeatmapID], list[Feature], np.ndarray]
BacdiveMetadata = dict[str, dict[str, str]]
_BacdiveJSONEntry = dict[str, dict]


class BacdiveAPISearcher:
    _singleton: "BacdiveAPISearcher | None" = None

    """
    Search BacDive database contents. Provide interface for acquiring data necessary to populate UI, such as
    PhylotreeDash component and summary heatmaps. Populate taxonomic annotations for given BacDive data.
    """

    def __init__(self, genetic_tools: GeneticToolDBList, bacdive_zj: Path, summary_parquet: Path):
        """
        Create API searcher using .zj zipped data and taxonomy summary files
        (created by the ``generate_bacdive_summary`` module)

        :param genetic_tools: List of genetic tool dbs
        :param bacdive_zj: Path to .zj summary file
        :param summary_parquet: Path to .parquet summary file
        """
        assert bacdive_zj.exists()
        self._logger = logging.getLogger(__name__)

        self._logger.info("Loading BacDive data from %s", bacdive_zj)
        with open(bacdive_zj, "r") as ptr:
            self._data: dict[str, _BacdiveJSONEntry] = json_unzip(json.load(ptr))

        self._logger.info("Scanning BacDive parquet summary from %s", summary_parquet)
        self._summary_data = pl.scan_parquet(summary_parquet)

        self._genetic_tools = genetic_tools
        self._logger.info("db loaded")

    @property
    def ranks(self) -> list[str]:
        """
        Ranks in BacDive taxonomic summary data.

        :return: List of taxonomic ranks
        """
        return ["domain", "phylum", "class", "order", "family", "genus", "species"]

    @property
    def ranks_with_strain(self) -> list[str]:
        """
        Ranks in BacDive taxonomic summary data, including strain designation

        :return: List of taxonomic ranks
        """
        return ["domain", "phylum", "class", "order", "family", "genus", "species", "strain"]

    @property
    def summary_df(self) -> pl.LazyFrame:
        """
        Reference to BacDive summary

        :return: LazyFrame
        """
        return self._summary_data

    def filter_by_species_taxid(self, taxids: Collection[int]) -> pl.LazyFrame:
        """
        Filter stored data by NCBI species tax ID

        :param taxids:  List of taxids to collect
        :return: LazyFrame
        """
        return self._summary_data.filter(pl.col("species-taxid").is_in(taxids))

    def filter_by_bacdive(self, bacdive_ids: Collection[BacdiveID]) -> pl.LazyFrame:
        """
        Filter stored data by BacDive ID

        :param bacdive_ids: BacDive IDs on which to filter
        :return: Filtered DataFrame
        """
        return self._summary_data.filter(pl.col("ID").is_in(bacdive_ids))

    @cached_property
    def taxids_with_traits(self) -> set[TaxID]:
        """
        Obtain all NCBI tax ids associated with BacDive entries

        :return: List of tax ids
        """
        return set(self._summary_data.select(pl.col("species-taxid").unique()).collect()["species-taxid"])

    def get_taxonomy(self, taxids: Collection[TaxID] | None) -> pl.DataFrame:
        """
        Get taxonomic assignments for list of NCBI tax ids

        :param taxids: List of NCBI tax ids for which to collect BacDive taxonomy summary information
        :return: DataFrame of BacDive taxonomy results
        """
        if taxids is None:
            return self._summary_data.select("ID", "species-taxid", *self.ranks_with_strain).fill_null(".").collect()
        return (
            self._summary_data.filter(pl.col("species-taxid").is_in(taxids))
            .select("ID", "species-taxid", *self.ranks_with_strain)
            .fill_null(".")
        ).collect()

    def filter_by_ranks(self, expected_tax: Sequence[str]) -> pl.LazyFrame:
        """
        Collect BacDive isolate data corresponding to entries whose taxonomy matches the provided list of taxonomic
        ranks

        :param expected_tax: List of taxonomic terms, max length == ``len(self.ranks)``
        :return: LazyFrame of matching data
        """
        assert 0 < len(expected_tax) <= len(self.ranks)
        ranks = self.ranks[: len(expected_tax)]
        expr: pl.Expr = pl.col(ranks[0]).eq(expected_tax[0])
        for rank, _path in zip(ranks[1:], expected_tax[1:], strict=True):
            expr = expr & pl.col(rank).eq(_path)
        return self._summary_data.filter(expr).select("ID", "species-taxid", *self.ranks)

    def _add_rings(
        self, leaves: list[int], out: defaultdict, mapping: dict[int, str], chatbot_provided: list[ChatbotPayload]
    ) -> None:
        leaves_set = set(leaves)
        for tool_db in self._genetic_tools:
            tool_df = tool_db.match(leaves).collect()
            for species_id in tool_df[tool_db.host_id_col].unique():
                out[mapping[species_id]]["traits"]["rings"].append(1)
            for unmatched in leaves_set.difference(tool_df[tool_db.host_id_col]):
                out[mapping[unmatched]]["traits"]["rings"].append(0)
        for entry in chatbot_provided:
            if entry.get("species") is None:
                continue
            if entry["species"] not in out.keys():
                out[entry["species"]] = {"traits": {"rings": [0, 0, 0]}}
            out[entry["species"]]["traits"]["rings"].append(1)

    def metadata(
        self, tree: PhyloNode, mapping: dict[int, str], chatbot_provided: list[ChatbotPayload] | None = None
    ) -> BacdiveMetadata:
        """
        Collect metadata for leaves in a tree rooted at ``PhyloNode`` and with name mapping provided by ``mapping``.

        ``mapping`` is typically generated by NCBITaxa.get_taxid_translator(NAMES) and is provided to the caller
        to prevent use cases in which this mapping was already generated by the caller (and is thus doubly-calculated).

        :param tree: Tree generated by NCBITaxa and referencing BacDive data
        :param mapping: Mapping between NCBI taxids and the node names in the tree
        :return: Mapping between nodes in the tree and the BacDive database entries associated with the leaf BacDive IDs
        """
        if chatbot_provided is None:
            chatbot_provided = [{"results": []}]
        ncbi_client = get_ncbi_taxa()
        out = defaultdict(dict)
        valid_ranks = set(self.ranks)
        leaves: list[int] = []
        for node in tree.iter_leaves():
            name_int = int(node.name)
            leaves.append(name_int)
            ranks = ncbi_client.get_rank(ncbi_client.get_lineage(node.name))
            translator = ncbi_client.get_taxid_translator(ranks.keys())
            out_ptr = out[mapping[name_int]]
            for taxid, rank in ranks.items():
                if rank in valid_ranks:
                    out_ptr[rank] = translator[taxid]
            # Mark entries that have BacDive entries
            if name_int in self.taxids_with_traits:
                out_ptr["traits"] = {"size": 1}
            else:
                out_ptr["traits"] = {}
            # Set empty rows array
            out_ptr["traits"]["rings"] = []
        # Add genetic tool db mapping info
        self._add_rings(leaves, out, mapping, chatbot_provided)
        return dict(out)

    def get(self, bacdive_ids: Iterable[str], ncbi_client: NCBITaxa) -> BacdiveSummaryList:
        """
        Get BacDive database entries from associated IDs.

        :param bacdive_ids: List of BacDive IDs
        :param ncbi_client: NCBITaxa instance
        :return: List of BacDiveSummary objects corresponding to requested IDs
        """
        return BacdiveSummaryList(
            [
                BacdiveAPISearcher._parse_bacdive_retrieve_result(self._data.get(bacdive_id, {}), ncbi_client)
                for bacdive_id in bacdive_ids
            ]
        )

    @staticmethod
    def _parse_bacdive_retrieve_result(strain_info: dict[str, Any], ncbi_client: NCBITaxa) -> BacdiveSummary:
        """
        Parse BacDive entry.

        :param strain_info: Stored data associated with a given BacdiveID in the database
        :param ncbi_client: NCBITaxa instance
        :return: Parsed data
        """
        # Collect culture temperature information
        _culture_info = strain_info.get("Culture and growth conditions", {})
        # Collect metabolite utilization information
        _metabolism_info = strain_info.get("Physiology and metabolism", {})
        metabolites = MetaboliteUtilization.parse_metabolites(_metabolism_info.get("metabolite utilization", []))
        # Collect detected enzyme activity
        enzymes = EnzymeActivity.parse_enzymes(_metabolism_info.get("enzymes", []))
        # Collect culture medium information
        culture_mediums = CultureMedium.parse_culture_media(_culture_info.get("culture medium", []))
        # Collect taxonomy information
        _general_info = strain_info.get("General", strain_info)  # Pre-parsed information has tax info at top-level
        ncbi_tax = NCBITax.parse_tax_list(
            _general_info.get("NCBI tax id", {}), BacdiveEntryRefID(_general_info.get("@ref", -1)), ncbi_client
        )
        # Get BacDive-assigned reference number
        ref = BacdiveRef(_general_info.get("BacDive-ID", -1))
        papers = PaperReference.parse_paper_references(strain_info.get("External links", {}))
        culture_temps = CultureTemperature.parse_culture_temps(_culture_info.get("culture temp", {}))
        return BacdiveSummary(
            ref=ref,
            description=_general_info.get("description", ""),
            ncbi_tax=ncbi_tax,
            culture_mediums=culture_mediums,
            culture_temps=culture_temps,
            metabolites=metabolites,
            enzymes=enzymes,
            papers=papers,
        )

    @staticmethod
    def get_searcher() -> "BacdiveAPISearcher":
        if BacdiveAPISearcher._singleton is None:
            BacdiveAPISearcher._singleton = BacdiveAPISearcher(
                load_genetic_tools(),
                Path("data") / "bacdive-all.summary.zj",
                Path("data") / "bacdive-taxonomy.parquet",
            )
        return BacdiveAPISearcher._singleton


def load_bacdive_data() -> pl.DataFrame:
    return (
        BacdiveAPISearcher.get_searcher()
        .get_taxonomy(None)
        .filter(pl.col("species-taxid").is_not_null() & pl.col("domain").eq("Bacteria"))
    )


if __name__ == "__main__":
    ...
