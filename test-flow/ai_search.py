import os
import json
from typing import List

from promptflow.core import tool
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_core.documents.base import Document
from dotenv import load_dotenv

from utils import get_search_results


load_dotenv()

# AI search configuration
indexes = [os.environ['AZURE_SEARCH_INDEX']]
aoai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
aoai_resources: list = json.loads(os.environ.get("AUZRE_OPENAI_RESOURCES"))
embed_llms: list[AzureOpenAIEmbeddings] = []
if aoai_resources is not None:
    for resource in aoai_resources:
        if resource["model_name"] == "embed":
            embed_llms.append(AzureOpenAIEmbeddings(
                openai_api_type="azure",
                openai_api_base=resource["endpoint"],
                openai_api_key=resource["key"],
                openai_api_version=aoai_api_version,
                deployment=resource["deployment_name"],
                chunk_size=1
            ))


@tool
def ai_search(query_text: str, k: int = 3) -> List[Document]:
    embedder = embed_llms[0]
    ordered_results = get_search_results(query_text,
                                         indexes,
                                         sas_token=os.environ.get("BLOB_SAS_TOKEN"),
                                         vector_search=True,
                                         similarity_k=k,
                                         query_vector=embedder.embed_query(query_text),
                                         enable_site_id=False)
    top_docs = []
    for key, value in ordered_results.items():
        location = value["location"] if value["location"] is not None else ""
        top_docs.append(
            Document(page_content=value["chunk"], metadata={"source": location}))
    return top_docs
