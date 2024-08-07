from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class EnumCitation(str, Enum):
    defaut = "default"
    markdoc = "markdoc"  # https://markdoc.dev/docs/tags


class Split(BaseModel):
    context_id: str = Field(description="Context where the split belongs to")
    content: str = Field(description="Text of the split")
    id: str = Field(description="Identification of the split within this message. Used to match citations.")
    split_number: int = Field(description="Where in the document is the split (splits are numbered sequentially inside the document, starting in 1)")
    split_id: str = Field(description="Unique ID for the split")
    document_title: str = Field(description="Title of the document where the split belongs to")
    document_description: str = Field(description="Description of the document where the split belongs to")
    filepath: str = Field(description="Filepath of the document ingested into the system (probably meaningless to the user)")
    page: int = Field(description="Page of the document where the split belongs to")
    url: str = Field(description="URL that points to the original document")


class OpenaiCompletionResponse(BaseModel):
    choices: list
    prompt_filter_results: list
    usage: dict
    citations: List[Split]
