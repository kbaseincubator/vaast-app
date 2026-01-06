"""
Chatbot interaction and processing module.

This module provides data models and helper functions for handling chatbot interactions,
specifically focused on genetic tools and taxonomy queries. It includes functionality
for processing chat requests, classifying question types (genetic tools vs. taxonomy),
retrieving related species and subspecies from NCBI taxonomy, and extracting structured
data for visualization.
"""

import json
import logging
import os
from io import StringIO
from textwrap import dedent
from typing import Literal, NewType, TypeVar

import polars as pl
from ete3 import NCBITaxa
from langchain_anthropic import ChatAnthropic
from openai._client import OpenAI
from paperqa import Docs, Settings
from paperqa.settings import AgentSettings
from pydantic import BaseModel, Field, ValidationError
from pydantic.types import SecretStr

logging.basicConfig(encoding="utf-8", level=logging.INFO)
TaxID = NewType("TaxID", int)
NonSpeciesEntry = NewType("NonSpeciesEntry", str)
T = TypeVar("T")


class ChatMessage(BaseModel):
    """Structure of message from chatbot"""

    sender: str
    message: str
    message_type: Literal["text", "table"]
    selection: list[str]
    rating: str | None


class ChatRequest(BaseModel):
    """Packaged recent message and chat history"""

    message: str
    message_type: Literal["text", "table"]
    selection: list[str]
    chat_history: list[ChatMessage]


class Tool(BaseModel):
    """Model representing a genetic tool found in literature"""

    type: str = Field(description="Type of genetic tool, such as promoter, plasmid, phage, etc.")
    name: str = Field(description="Name of tool entity, such as 'Orf211'")
    references: list[str] = Field(description="DOI reference to publication that describes tool")


class ToolResponse(BaseModel):
    """
    Expected JSON output for model

    '"species": "", "tools": [{"type": "", "name": "", "references": []}]'
    """

    species: str = Field(description="Name of bacterial species, such as Pseudomonas aeruginosa")
    tools: list[Tool] = Field(description="List of genetic tools describing a given bacterial entity")


class BacteriaSummary(BaseModel):
    """
    Summary of bacterial tools and species found in text
    """

    bacteria: list[ToolResponse] = Field(min_length=1)


class QuestionType(BaseModel):
    """
    Classifying the type of question asked by the user
    """

    type: Literal["genetic tool", "taxonomy", "not relevant"]


class TaxonomyQuestionType(BaseModel):
    """
    Classifying the type of taxonomy-related question asked by the user
    """

    type: Literal["nomenclature", "related species", "subspecies or strains", "other"]
    species: list[str]


def _get_related_species(tax_names: list[str], species: list[str]) -> list[dict[str, str]]:
    """
    Retrieve related species sharing the same parent taxon for the given inputs.

    Queries the NCBI taxonomy database to find the parent taxon (e.g., genus) of the
    provided taxonomic names and returns a list of all descendant species.

    Args:
        tax_names: A list of taxonomic names to search.
        species: A list of species names to search.

    Returns:
        A list of dictionaries with 'taxid' and 'name' keys for the related species.

    """
    query = tax_names + species
    ncbi_client = NCBITaxa()
    translator = ncbi_client.get_name_translator(query)
    query_taxids = []
    for query_tax in query:
        taxids = translator.get(query_tax, None)
        # Not found, or described uncharacterized sequences
        if not taxids or 32644 in taxids:
            continue
        # Is a root-level identifier or not Bacteria
        lineage = ncbi_client.get_lineage(taxids[0]) or []
        if taxids[0] < 5 or not lineage or 2 not in lineage:
            continue
        # Confirm entry is species-level
        rank = ncbi_client.get_rank([taxids[0]]).get(taxids[0], "")
        if rank == "species":
            query_taxids.append(lineage[-2])

    return [
        {"taxid": tax[0], "name": tax[1]}
        for taxid in query_taxids
        for tax in ncbi_client.get_taxid_translator(ncbi_client.get_descendant_taxa(taxid)).items()
    ]


def _get_subspecies(tax_names: list[str], species: list[str]) -> list[dict[str, str]]:
    """
    Retrieve subspecies or strains for the given inputs.

    Queries the NCBI taxonomy database to find the descendants of the
    provided taxonomic names and returns a list of all descendant subspecies
    or strains.

    Args:
        tax_names: A list of taxonomic names to search.
        species: A list of species names to search.

    Returns:
        A list of dictionaries with 'taxid' and 'name' keys for the subspecies/strains.

    """
    query = tax_names + species
    ncbi_client = NCBITaxa()
    translator = ncbi_client.get_name_translator(query)
    query_taxids = []
    for query_tax in query:
        taxids = translator.get(query_tax, None)
        # Not found, or described uncharacterized sequences
        if not taxids or 32644 in taxids:
            continue
        descendents = [
            {"taxid": tax, "name": name}
            for taxid in taxids
            for tax, name in ncbi_client.get_taxid_translator(ncbi_client.get_descendant_taxa(taxid)).items()
        ]
        query_taxids.extend(descendents)
    return query_taxids


def _search_names_parquet(names: list[str]) -> list[dict[str, str]]:
    """
    Search for scientific names in the NCBI taxonomy parquet file.

    Scans the local parquet file for scientific names matching the input list.
    Verifies that the matched names belong to the Bacteria kingdom (taxid 2).

    Args:
        names: A list of partial or full names to search for.

    Returns:
        A list of dictionaries with 'taxid' and 'name' keys for the matching species.

    """
    names_pkt = pl.scan_parquet("data/ncbi-tax-names.parquet")
    ncbi_client = NCBITaxa()
    out = []
    for name in names:
        # Collect all taxids with entries that start with the query name
        entry_data = names_pkt.filter(pl.col("name_txt").str.contains(name)).select("tax_id").collect()
        # Get taxid and names of entries that are in `entry_data` and that are "scientific name" entry types
        sci_names = (
            names_pkt.filter(
                (pl.col("tax_id").is_in(entry_data["tax_id"]) & pl.col("name class").eq("scientific name"))
            ).select("tax_id", "name_txt")
        ).collect()
        potential_entries = (sci_names["tax_id"].unique().to_list(), sci_names["name_txt"].unique().to_list())
        for taxid, name_txt in zip(*potential_entries, strict=True):
            try:
                if 2 not in (ncbi_client.get_lineage(taxid) or []):
                    continue
            except ValueError:
                continue
            out.append({"taxid": taxid, "name": name_txt})
    return out


def _taxonomy_question(question_type: str, chat_request: ChatRequest, species: list[str]) -> ChatRequest:
    """
    Handle taxonomy-related questions and return a structured response.

    Router function that dispatches the request to the appropriate handler based
    on the specific type of taxonomy question (nomenclature, related species,
    or subspecies/strains).

    Args:
        question_type: The specific type of taxonomy question.
        chat_request: The original chat request object.
        species: A list of relevant species names associated with the question.

    Returns:
        ChatRequest: A new ChatRequest object containing the processed response as a JSON string.

    Raises:
        ValueError: If an unknown question_type is provided.

    """
    if question_type == "nomenclature":
        return ChatRequest(
            selection=chat_request.selection,
            message=json.dumps(_search_names_parquet(chat_request.selection + species)),
            chat_history=chat_request.chat_history,
            message_type="table",
        )
    if question_type == "related species":
        return ChatRequest(
            selection=chat_request.selection,
            message=json.dumps(_get_related_species(chat_request.selection, species)),
            chat_history=chat_request.chat_history,
            message_type="table",
        )
    if question_type == "subspecies or strains":
        return ChatRequest(
            selection=chat_request.selection,
            message=json.dumps(_get_subspecies(chat_request.selection, species)),
            chat_history=chat_request.chat_history,
            message_type="table",
        )
    raise ValueError("unreachable code")


class ChatbotClient:
    # TODO: Add ollama support
    def __init__(
        self,
        docs: Docs,
        version: Literal["OpenAI", "Anthropic"],
        hosting_location: Literal["API", "CBORG"],
        model: str,
    ):
        """
        Initialize the ChatbotClient with the specified configuration.

        Args:
            docs: The paperqa Docs object for querying.
            version: The LLM provider version ('OpenAI' or 'Anthropic').
            hosting_location: The hosting environment ('API' or 'CBORG').
            model: The name of the LLM model to use.

        """
        self._docs = docs
        self._version = version
        self._host = hosting_location
        self._model = model

    def _get_settings(self, prompt: str | None = None) -> Settings:
        # local_llm_config = dict(
        #     model_list=[
        #         dict(
        #             model_name="ollama/gemma2-32k",
        #             litellm_params=dict(
        #                 model="ollama/gemma2-32k",
        #                 api_base=_API_BASE,
        #             ),
        #         )
        #     ]
        # )
        # settings = Settings(
        #     llm="ollama/gemma2-32k",
        #     llm_config=local_llm_config,
        #     summary_llm="ollama/gemma2-32k",
        #     summary_llm_config=local_llm_config,
        #     embedding="ollama/snowflake-arctic-embed2",
        #     agent=AgentSettings(agent_llm="ollama/gemma2-32k", agent_llm_config=local_llm_config),
        # )
        match self._version:
            case "OpenAI":
                settings = Settings(
                    llm=self._model,
                    summary_llm=self._model,
                    agent=AgentSettings(agent_llm=self._model),
                )
            case "Anthropic":
                settings = Settings(
                    llm=self._model,
                    summary_llm=self._model,
                    agent=AgentSettings(agent_llm=self._model),
                    embedding="st-multi-qa-MiniLM-L6-cos-v1",
                )
            case _:
                raise ValueError("unreachable code")
        settings.answer.answer_max_sources = 10
        settings.answer.evidence_k = 5
        if prompt:
            settings.prompts.qa = prompt
        return settings

    def generate_response(self, prompt: str, return_type: type[T]) -> T:
        """
        Generate a structured response from the LLM.

        Sends a prompt to the configured LLM backend and parses the response
        into the specified Pydantic model type.

        Args:
            prompt: The prompt text to send to the LLM.
            return_type: The Pydantic model class expected for the response.

        Returns:
            An instance of the specified return_type containing the structured data.

        """
        match self._version:
            case "OpenAI":
                return self._generate_response_api_openai(prompt, return_type)
            case "Anthropic":
                return self._generate_response_api_anthropic(prompt, return_type)
            case _:
                raise ValueError("unreachable code")

    def _generate_response_api_openai(self, prompt: str, return_type: type[T]) -> T:
        if self._host == "API":
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        else:
            client = OpenAI(api_key=os.environ.get("CBORG_API_KEY", ""), base_url="https://api.cborg.lbl.gov")
        response = client.chat.completions.parse(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            response_format=return_type,
            reasoning_effort="medium",
        )
        result = response.choices[0].message.parsed
        if not result:
            raise RuntimeError("failed to generate response")
        return result

    def _generate_response_api_anthropic(self, prompt: str, return_type: type[T]) -> T:
        client = ChatAnthropic(
            api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY" if self._host == "API" else "CBORG_API_KEY", "")),
            model_name=self._model,
            temperature=1,
            timeout=None,
            max_retries=2,
            stop=None,
            max_tokens_to_sample=24000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            **(dict(base_url="https://api.cborg.lbl.gov") if self._host == "CBORG" else {}),
        ).bind_tools([return_type])
        response = client.invoke([{"role": "user", "content": prompt}])

        if not response.tool_calls or not response.tool_calls[0]["args"]:
            raise ValueError("no tools (i.e., Pydantic types) were called")

        try:
            return return_type.model_validate(response.tool_calls[0]["args"])
        except ValidationError as e:
            print(e)
            raise

    def process_chat(self, chat_request: ChatRequest) -> ChatRequest:
        """
        Process a chat request and return a response.

        Determines if the question is about genetic tools or taxonomy and routes it accordingly.

        Args:
            chat_request: The user's chat request.

        Returns:
            ChatRequest: The assistant's response.

        """

        def _default_err_response(chat_request: ChatRequest, msg: str = "genetic tools and taxonomy") -> ChatRequest:
            return ChatRequest(
                message=f"Hmmm, I currently only respond to questions about {msg}. Please ask a different question",
                chat_history=chat_request.chat_history,
                selection=chat_request.selection,
                message_type="text",
            )

        # Get question type
        try:
            response = self.generate_response(
                prompt=(
                    f"A user is asking the following question: '{chat_request.message}'.\n"
                    "Is this question related to biomanufacturing and genetic engineering (like antibiotics, phages, plasmids, CRISPR, cassettes, etc.), "
                    "taxonomy (like alternative nomenclature, near relatives to a given organism, etc.), "
                    "or is it not relevant?"
                ),
                return_type=QuestionType,
            )
        except BaseException:
            return _default_err_response(chat_request)

        if response.type == "genetic tool":
            chat_history = StringIO()
            for v in chat_request.chat_history:
                chat_history.write(f"{v.sender}: bacteria: {','.join(v.selection)} message: {v.message}\n")
            if not chat_request.selection:
                region = (
                    "Your colleage is studying genetic tools across all known bacterial hosts and asked the "
                    f"question '{chat_request.message}'\n"
                )
            else:
                region = (
                    f"Your colleage is studying the bacteria ({','.join(chat_request.selection)}) "
                    f"and asked the question '{chat_request.message}'\n"
                )

            prompt = (
                "You are a synthetic biology researcher who is having a conversation with a colleague.\n"
                f"{region}"
                "Respond to their comment/question with specific sources AND the DOI of the manuscript. "
                "You must provide responses that include actual tool entries. Be as comprehensive as possible.\n"
                "If the user does not specify a bacterial organism in their question, assume that they are referring to the organisms in the first part of their conversation.\n"
                "***********************************************"
                "Here is the first part of the conversation:\n\n"
                f"{chat_history.getvalue()}\n\n"
                "***********************************************"
                "Use the context below if helpful. If you come across any names like 'E. coli', translate these to the correct "
                "scientific name, like Escherichia coli. Always include the full organism name in every response you make to the user.\n"
                "Context: {context}\n"
                "Response: "
            )
            answer = self._docs.query(chat_request.message, settings=self._get_settings(prompt))
            return ChatRequest(
                message=answer.answer,
                chat_history=chat_request.chat_history,
                selection=chat_request.selection,
                message_type="text",
            )

        if response.type == "taxonomy":
            try:
                response = self.generate_response(
                    prompt=(
                        f"A user is asking the following question about bacterial taxonomy: '{chat_request.message}'.\n"
                        "Are they asking about nomenclature (such as alternative names for the organism), "
                        "for a list of related organimsms (such as related species), "
                        "for a list of subspecies or strains, "
                        "or something else (other)? Determine the question type, and extract any relevant bacterial species from their question"
                    ),
                    return_type=TaxonomyQuestionType,
                )
            except BaseException:
                return _default_err_response(chat_request, "nomenclature changes, related organisms, and subspecies")

            if response.type == "not relevant":
                return _default_err_response(chat_request, "nomenclature changes, related organisms, and subspecies")
            return _taxonomy_question(response.type, chat_request, response.species)

        return _default_err_response(chat_request)

    def visualizer_data(self, chat_request: ChatRequest) -> list[ToolResponse]:
        """
        Extract visualizer data from a chat request.

        Uses an LLM to parse the latest chat message and extract structured
        information about bacteria and their associated genetic tools for
        visualizer display.

        Args:
            chat_request: The request containing chat history and user selection.

        Returns:
            list[ToolResponse]: A list of extracted bacterial tool information.

        """

        def get_bacteria_summary(
            text: str, selection: list[str], example_summaries: list[tuple[str, BacteriaSummary]]
        ) -> str:
            prompt = dedent(
                """
                You will extract summarized information about bacterial entities in the following paragraphs.
                Look first for descriptions of bacteria that describe phages, plasmids, CRISPR elements, and other genetic
                toolsets. If there is more information about the bacteria after looking for these elements, provide this data, too.
                ***Only include examples that have BOTH a bacterial entity AND one or more tool entities. 
                If you come across any names like 'E. coli', translate these to the correct scientific name, like Escherichia coli.

            
                ==== EXAMPLES ====
                %s
                ==================

                Describe the bacteria in the following paragraphs in JSON format:
                ==== PROMPT ====
                bacteria: %s
                message: %s
                ================
                """
            )
            summary_region = StringIO()
            for example in example_summaries:
                summary_region.write(f"PARAGRAPH:\n{example[0]}\nRESULT:\n{example[1].model_dump_json()}\n\n")

            return prompt % (summary_region.getvalue(), ",".join(selection), text)

        examples = [
            (
                "Escherichia coli hasa a number of phage viruses which can infect it. These were described in "
                "Park et. al. 2015 and include broad infections like lambda virus and more specific diseases caused by"
                "T4 viruses.",
                BacteriaSummary(
                    bacteria=[
                        ToolResponse(
                            species="Escherichia coli",
                            tools=[
                                Tool(name="T4", type="phage", references=["Park et. al. 2015"]),
                                Tool(name="lambda", type="phage", references=["Park et. al. 2015"]),
                            ],
                        )
                    ]
                ),
            ),
            (
                "The genus Pseudomonas is a large and diverse group of bacteria, including many species that are capable of "
                "degrading a wide range of complex organic molecules.  Some examples include Pseudomonas putida, which can "
                "degrade a variety of aromatic compounds such as benzene, toluene, and xylene, as well as the aliphatic "
                "compound chloroform.  Pseudomonas aeruginosa is another species that is capable of degrading a wide range of "
                "organic compounds, including the aromatic compounds naphthalene and biphenyl, as well as the aliphatic "
                "compound 1,2-dichloroethane.  Overall, the ability of Pseudomonas species to degrade complex organic "
                "molecules is due to the presence of specialized enzymes called cytochrome P450 enzymes, which are capable of"
                "catalyzing the oxidation of a wide range of organic compounds.",
                BacteriaSummary(
                    bacteria=[
                        ToolResponse(
                            species="Pseudomonas putida",
                            tools=[
                                Tool(name="benzene degradation", type="metabolic", references=[]),
                                Tool(name="toluene degradation", type="metabolic", references=[]),
                                Tool(name="xylene degradation", type="metabolic", references=[]),
                            ],
                        ),
                        ToolResponse(
                            species="Pseudomonas aeruginosa",
                            tools=[
                                Tool(name="biphenyl degradation", type="metabolic", references=[]),
                                Tool(name="naphthalene degradation", type="metabolic", references=[]),
                                Tool(name="1,2-dichloroethane degradation", type="metabolic", references=[]),
                            ],
                        ),
                    ]
                ),
            ),
        ]

        return self.generate_response(
            get_bacteria_summary(
                chat_request.chat_history[-1].message,
                chat_request.chat_history[-1].selection,
                examples,
            ),
            BacteriaSummary,
        )
