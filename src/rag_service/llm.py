from functools import lru_cache

import anthropic
from pydantic import BaseModel, Field

from .config import get_settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical name of the entity, as written in the text")
    type: str = Field(description="One of: person, organization, location, product, technology, event, concept, other")
    description: str = Field(description="One sentence describing the entity based only on the text")


class ExtractedRelation(BaseModel):
    source: str = Field(description="Name of the source entity (must match an extracted entity name)")
    target: str = Field(description="Name of the target entity (must match an extracted entity name)")
    type: str = Field(description="Short UPPER_SNAKE_CASE relation label, e.g. WORKS_AT, PART_OF, USES")
    description: str = Field(description="One sentence explaining the relationship as stated in the text")


class GraphExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


EXTRACTION_SYSTEM = """You build a knowledge graph from document chunks.
Extract the salient entities in the chunk and the relationships between them that the text states or
clearly implies. Prefer specific, reusable entity names (e.g. "PostgreSQL", not "the database") so the
same entity merges across chunks. Skip generic nouns that would not help answer questions.
Every relation's source and target must be names from your entity list. Return empty lists if the chunk
has nothing worth extracting."""

ANSWER_SYSTEM = """You answer questions using only the retrieved context provided by a retrieval system:
text chunks from indexed documents and knowledge-graph facts linking entities.
Cite chunks you rely on inline as [C1], [C2], ... and graph facts as [G1], [G2], ...
If the context does not contain the answer, say so plainly instead of guessing."""


@lru_cache
def get_client() -> anthropic.Anthropic:
    key = get_settings().anthropic_api_key
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _fallback_kwargs() -> dict:
    if get_settings().llm_fallbacks:
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return {}


class LLMRefusal(RuntimeError):
    pass


def extract_graph(chunk_text: str, document_title: str) -> GraphExtraction:
    s = get_settings()
    response = get_client().beta.messages.parse(
        model=s.extraction_model,
        max_tokens=16000,
        system=EXTRACTION_SYSTEM,
        output_config={"effort": s.extraction_effort},
        output_format=GraphExtraction,
        messages=[
            {
                "role": "user",
                "content": f"<document_title>{document_title}</document_title>\n<chunk>\n{chunk_text}\n</chunk>",
            }
        ],
        **_fallback_kwargs(),
    )
    if response.stop_reason == "refusal":
        raise LLMRefusal(getattr(response.stop_details, "explanation", None) or "extraction refused")
    if response.parsed_output is None:
        raise RuntimeError(f"extraction returned no parsed output (stop_reason={response.stop_reason})")
    return response.parsed_output


def generate_answer(question: str, context: str) -> str:
    s = get_settings()
    with get_client().beta.messages.stream(
        model=s.llm_model,
        max_tokens=32000,
        system=ANSWER_SYSTEM,
        output_config={"effort": s.llm_effort},
        messages=[
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\n<question>{question}</question>",
            }
        ],
        **_fallback_kwargs(),
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise LLMRefusal(getattr(message.stop_details, "explanation", None) or "answer refused")
    return "".join(b.text for b in message.content if b.type == "text")
