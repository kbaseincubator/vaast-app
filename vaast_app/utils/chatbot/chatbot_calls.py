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
from typing import Literal, NewType, TypeVar, cast

import polars as pl
from ete3 import NCBITaxa
from langchain_anthropic import ChatAnthropic
from ollama import chat
from openai._client import OpenAI
from paperqa import Docs, Settings
from paperqa.settings import AgentSettings
from pydantic import BaseModel, Field, ValidationError
from pydantic.types import SecretStr

logging.basicConfig(encoding="utf-8", level=logging.INFO)
TaxID = NewType("TaxID", int)
NonSpeciesEntry = NewType("NonSpeciesEntry", str)
T = TypeVar("T", BaseModel, None)
logger = logging.getLogger(__name__)


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

    type: Literal["taxonomy", "genetic engineering", "follow up question", "not relevant"]


class TaxonomyQuestionType(BaseModel):
    """
    Classifying the type of taxonomy-related question asked by the user
    """

    type: Literal["nomenclature", "related species", "subspecies or strains", "other"]
    species: list[str]


class ResponseType(BaseModel):
    """
    Helper class for a simple response to an existing question
    """

    response: str


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
    logger.info("Scanning parquet file: data/ncbi-tax-names.parquet for %s", names)

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


class InvalidOllamaEmbeddingsError(Exception):
    """
    Exception raised when invalid embeddings are specified for Ollama host.
    """


class InvalidHostingLocationError(Exception):
    """
    Exception raised when both use_cborg and use_ollama are True.
    """


class ChatbotClient:
    """
    Client for handling chatbot interactions and LLM integration.

    This class manages the connection to LLM providers (OpenAI, Anthropic),
    handles query processing, and routes requests to appropriate handlers
    based on question type (taxonomy vs. genetic tools).
    """

    def __init__(
        self,
        docs: Docs,
        version: Literal["OpenAI", "Anthropic"],
        model: str,
        use_cborg: bool = False,
        use_ollama: bool = False,
        embeddings: str | None = None,
    ):
        """
        Initialize the ChatbotClient with the specified configuration.

        Args:
            docs: The paperqa Docs object for querying.
            version: The LLM provider version ('OpenAI' or 'Anthropic').
            model: The name of the LLM model to use.
            use_cborg: Whether to use CBORG as the hosting location.
            use_ollama: Whether to use Ollama as the hosting location.
            embeddings: The name of the LLM embeddings to use (required when using 'Ollama' host)

        Raises:
            InvalidHostingLocationError: If both use_cborg and use_ollama are True.
            InvalidOllamaEmbeddingsError: If embeddings are not specified when use_ollama is True.

        """
        self._docs = docs
        self._version = version
        self._use_cborg = use_cborg
        self._use_ollama = use_ollama
        self._model = model
        self._embeddings = embeddings
        if self._use_cborg and self._use_ollama:
            raise InvalidHostingLocationError("Both use_cborg and use_ollama are True")
        if self._use_ollama and not self._embeddings:
            raise InvalidOllamaEmbeddingsError("Embeddings must be specified when using Ollama host")

    def _get_settings(self, prompt: str | None = None) -> Settings:
        """
        Configure and retrieve the settings for the LLM based on hosting location and provider.

        Constructs the Settings object required by paperqa, configuring the LLM model,
        hosting location (API or CBORG), and provider-specific parameters (OpenAI vs Anthropic).
        It also sets up agent settings and search parameters.

        Args:
            prompt: Optional custom prompt to override the default QA prompt.

        Returns:
            Settings: A configured paperqa Settings object.

        Raises:
            ValueError: If the hosting location or version is invalid.

        """
        if self._use_ollama or self._use_cborg:
            additional_data = {
                "model_list": [
                    {
                        "model_name": self._model,
                        "litellm_params": {
                            "model_name": self._model,
                            "model": self._model,
                            **(
                                {
                                    "api_base": (
                                        "http://127.0.0.1:11434" if self._use_ollama else "https://api.cborg.lbl.gov"
                                    ),
                                    "api_key": os.environ.get("CBORG_API_KEY", ""),
                                }
                            ),
                        },
                    }
                ]
            }
            to_add = dict(llm_config=additional_data, summary_llm_config=additional_data)
        else:
            additional_data = None
            to_add = {}
        match self._version:
            case "OpenAI":
                additional_setting = {}
            case "Anthropic":
                additional_setting = {"embedding": "st-multi-qa-MiniLM-L6-cos-v1"}
            case "Ollama":
                raise NotImplementedError()
            case _:
                raise ValueError("unreachable code")
        settings = Settings(
            llm=self._model,
            summary_llm=self._model,
            agent=AgentSettings(
                agent_llm=self._model,
                timeout=1000,
                # pyrefly: ignore
                **({"agent_llm_config": additional_data} if additional_data else {}),
            ),
            prompts={"use_json": False},
            # pyrefly: ignore
            **({"embedding": self._embeddings} if self._use_ollama else {}),
            **additional_setting,
            **to_add,
        )

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
        if not self._use_cborg:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        else:
            client = OpenAI(api_key=os.environ.get("CBORG_API_KEY", ""), base_url="https://api.cborg.lbl.gov")

        logger.info("Sending request to OpenAI model: %s. Prompt length: %d", self._model, len(prompt))
        try:
            response = client.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                response_format=return_type,
                reasoning_effort="medium",
            )
            result = response.choices[0].message.parsed
        except Exception as e:
            logger.exception("Error generating response from OpenAI")
            raise e

        if not result:
            logger.error("Failed to generate response from OpenAI (empty result)")
            raise RuntimeError("failed to generate response")

        logger.info("Received successful response from OpenAI")
        return result

    def _generate_response_api_anthropic(self, prompt: str, return_type: type[T]) -> T:
        client = ChatAnthropic(
            api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY" if not self._use_cborg else "CBORG_API_KEY", "")),
            model_name=self._model,
            temperature=1,
            timeout=None,
            max_retries=2,
            stop=None,
            max_tokens_to_sample=24000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            # pyrefly: ignore
            **(dict(base_url="https://api.cborg.lbl.gov") if self._use_cborg else {}),
        ).bind_tools([return_type])

        logger.info("Sending request to Anthropic model: %s. Prompt length: %d", self._model, len(prompt))
        try:
            response = client.invoke([{"role": "user", "content": prompt}])

            if not response.tool_calls or not response.tool_calls[0]["args"]:
                raise ValueError("no tools (i.e., Pydantic types) were called")

            result = cast(BaseModel, return_type).model_validate(response.tool_calls[0]["args"])
            logger.info("Received successful response from Anthropic")
            return cast(T, result)
        except ValidationError as e:
            logger.exception("Validation error in Anthropic response")
            raise e
        except Exception as e:
            logger.exception("Error generating response from Anthropic")
            raise e

    def _generate_response_ollama(self, prompt: str, return_type: type[T]) -> T:
        assert issubclass(return_type, BaseModel)
        # pyrefly: ignore
        response = chat(
            model=self._model,
            messages=prompt,
            format=return_type.model_json_schema(),
            think="medium",
        )

        # Validate and parse the response against the expected output type
        return cast(T, return_type.model_validate_json(cast(str, response.message.content)))

    def process_chat(self, chat_request: ChatRequest) -> ChatRequest:
        """
        Process a chat request and return a response.

        Determines if the question is about genetic tools or taxonomy and routes it accordingly.

        Args:
            chat_request: The user's chat request.

        Returns:
            ChatRequest: The assistant's response.

        """

        # TODO: Error response for invalid requests should display in UI
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
                    + (
                        f"in response to the following question: '{chat_request.chat_history[-1].message}'\n\n"
                        if len(chat_request.chat_history) > 1
                        else ""
                    )
                    + "Is this question related to taxonomy (like alternative nomenclature, near relatives to a given organism, etc.), "
                    "genetic engineering (like plasmid use, phage infection, antibiotic selection, growth conditions, and other topics) "
                    "a follow up question about the previous response, "
                    "or is it not relevant?"
                ),
                return_type=QuestionType,
            )
        except BaseException as err:
            logger.error(err)
            return _default_err_response(chat_request)

        if response.type == "genetic engineering":
            logger.info("question is likely about genetic engineering")
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
                # "***********************************************"
                # "Here is the first part of the conversation:\n\n"
                # f"{chat_history.getvalue()}\n\n"
                # "***********************************************"
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
            logger.info("question is likely about taxonomy")
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

            return _taxonomy_question(response.type, chat_request, response.species)

        if response.type == "follow up question":
            logger.info("question is a follow up question")
            chat_history = StringIO()
            for v in chat_request.chat_history:
                chat_history.write(f"{v.sender}: bacteria: {','.join(v.selection)} message: {v.message}\n\n")
            chat_history.write(
                f"user follow-up question: bacteria: {','.join(chat_request.selection)} message: {chat_request.message}\n\n"
            )
            prompt = (
                "You are a synthetic biology researcher who is having a conversation with a colleague.\n"
                "You will be provided with a conversation history and a follow-up question.\n"
                "Respond to their follow-up question.\n"
                "Use the context below if helpful."
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

        if response.type == "not relevant":
            logger.info("question is not relevant")
            return _default_err_response(chat_request, "nomenclature changes, related organisms, and subspecies")

        logger.info("question is not relevant")
        return _default_err_response(chat_request)

    def visualizer_data(self, chat_request: ChatRequest) -> BacteriaSummary:
        """
        Extract visualizer data from a chat request.

        Uses an LLM to parse the latest chat message and extract structured
        information about bacteria and their associated genetic tools for
        visualizer display.

        Args:
            chat_request: The request containing chat history and user selection.

        Returns:
            BacteriaSummary: A list of extracted bacterial tool information.

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
