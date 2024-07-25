import re
import os
import sys
import json
from io import BytesIO
import logging
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests

from collections import OrderedDict
import base64

import redis
import docx2txt
import tiktoken
import html
import time
from pypdf import PdfReader, PdfWriter
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from azure.monitor.opentelemetry import configure_azure_monitor

from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_csv_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
# from openai.error import AuthenticationError
from langchain.docstore.document import Document
from pypdf import PdfReader
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

try:
    from prompts import (COMBINE_QUESTION_PROMPT, COMBINE_PROMPT, COMBINE_CHAT_PROMPT,
                         CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX, MSSQL_PROMPT, MSSQL_AGENT_PREFIX,
                         MSSQL_AGENT_FORMAT_INSTRUCTIONS, CHATGPT_PROMPT, BING_PROMPT_PREFIX, DOCSEARCH_PROMPT_PREFIX)
except Exception as e:
    print(e)
    from prompts import (COMBINE_QUESTION_PROMPT, COMBINE_PROMPT, COMBINE_CHAT_PROMPT,
                         CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX, MSSQL_PROMPT, MSSQL_AGENT_PREFIX,
                         MSSQL_AGENT_FORMAT_INSTRUCTIONS, CHATGPT_PROMPT, BING_PROMPT_PREFIX, DOCSEARCH_PROMPT_PREFIX)


class SharePointCache:
   def __init__(self, expiration_time_in_s, host, port=6379):
      self.time_to_expire_s=expiration_time_in_s
      self.client = redis.Redis(host=host, port=6379)
   def set_key(self, key, value):
      self.client.set(key, json.dumps(value), ex=self.time_to_expire_s)
   def get_key(self, key):
      return self.client.get(key)

def text_to_base64(text):
    # Convert text to bytes using UTF-8 encoding
    bytes_data = text.encode('utf-8')

    # Perform Base64 encoding
    base64_encoded = base64.b64encode(bytes_data)

    # Convert the result back to a UTF-8 string representation
    base64_text = base64_encoded.decode('utf-8')

    return base64_text


def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in
            range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1:
                cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1:
                cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


def parse_pdf(file, form_recognizer=False, formrecognizer_endpoint=None, formrecognizerkey=None,
              model="prebuilt-document", from_url=False, verbose=False):
    """Parses PDFs using PyPDF or Azure Document Intelligence SDK (former Azure Form Recognizer)"""
    offset = 0
    page_map = []
    if not form_recognizer:
        if verbose:
            print(f"Extracting text using PyPDF")
        reader = PdfReader(file)
        pages = reader.pages
        for page_num, p in enumerate(pages):
            page_text = p.extract_text()
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)
    else:
        if verbose:
            print(f"Extracting text using Azure Document Intelligence")
        credential = AzureKeyCredential(os.environ["FORM_RECOGNIZER_KEY"])
        form_recognizer_client = DocumentAnalysisClient(endpoint=os.environ["FORM_RECOGNIZER_ENDPOINT"],
                                                        credential=credential)

        if not from_url:
            with open(file, "rb") as filename:
                poller = form_recognizer_client.begin_analyze_document(model, document=filename)
        else:
            poller = form_recognizer_client.begin_analyze_document_from_url(model, document_url=file)

        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if
                              table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1] * page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >= 0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

    return page_map


def read_pdf_files(files, form_recognizer=False, verbose=False, formrecognizer_endpoint=None, formrecognizerkey=None):
    """This function will go through pdf and extract and return list of page texts (chunks)."""
    text_list = []
    sources_list = []
    for file in files:
        page_map = parse_pdf(file, form_recognizer=form_recognizer, verbose=verbose,
                             formrecognizer_endpoint=formrecognizer_endpoint, formrecognizerkey=formrecognizerkey)
        for page in enumerate(page_map):
            text_list.append(page[1][2])
            sources_list.append(file.name + "_page_" + str(page[1][0] + 1))
    return [text_list, sources_list]


def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def text_to_docs(text: List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


# Returns the num of tokens used on a string
def num_tokens_from_string(string: str) -> int:
    encoding_name = 'cl100k_base'
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Returning the toekn limit based on model selection
def model_tokens_limit(model: str) -> int:
    """Returns the number of tokens limits in a text model."""
    if model == "gpt-35-turbo":
        token_limit = 4096
    elif model == "gpt-4":
        token_limit = 120000
    elif model == "gpt-35-turbo-16k":
        token_limit = 16384
    elif model == "gpt-4-32k":
        token_limit = 32768
    else:
        token_limit = 4096
    return token_limit


# Returns num of toknes used on a list of Documents objects
def num_tokens_from_docs(docs: List[Document]) -> int:
    num_tokens = 0
    for i in range(len(docs)):
        num_tokens += num_tokens_from_string(docs[i].page_content)
        num_tokens += num_tokens_from_string(docs[i].metadata["source"])
    return num_tokens


def get_search_results(query: str, indexes: list,
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "",
                       vector_search: bool = False,
                       similarity_k: int = 3,
                       query_vector: list = [],
                       site_ids: list = None, enable_site_id=True) -> List[dict]:
    headers = {'Content-Type': 'application/json', 'api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    agg_search_results = dict()

    for index in indexes:
        # construct search payload
        search_payload = {
            "search": query,
            "queryType": "semantic",
            "semanticConfiguration": "my-semantic-config",
            "count": "true",
            "captions": "extractive",
            "answers": "extractive",
            "top": k
        }
        log_search_payload = search_payload.copy()
        if enable_site_id:
            if site_ids:
                filter_query = ""
                for site_id in site_ids:
                    if site_id == site_ids[-1]:
                        filter_query = filter_query + f"metadata_spo_site_id eq '{site_id}'"
                    else:
                        filter_query = filter_query + f"metadata_spo_site_id eq '{site_id}' or "
                search_payload["filter"] = filter_query
            else:
                filter_query = "metadata_spo_site_id eq ''"
                search_payload["filter"] = filter_query
        if vector_search:
            search_payload["vectorQueries"] = [
                {"kind": "vector", "vector": query_vector, "fields": "chunkVector", "k": k}]
            search_payload["select"] = "id, title, chunk, location"
            log_search_payload["vectorQuery"] = True
        else:
            search_payload["select"] = "id, title, chunks, name, location, vectorized"

        log_search_payload["select"] = search_payload["select"]
        logger.debug(f"search payload: {log_search_payload}", extra=log_search_payload)

        url = os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search"
        resp = requests.post(url, data=json.dumps(search_payload), headers=headers, params=params, timeout=20)
        resp.raise_for_status()

        search_results = resp.json()
        agg_search_results[index] = search_results

    content = dict()
    ordered_content = OrderedDict()

    for index, search_results in agg_search_results.items():
        for result in search_results['value']:
            if result[
                    '@search.rerankerScore'] > reranker_threshold:  # Show results that are at least N% of the max possible score=4
                content[result['id']] = {
                    "title": result['title'],
                    "location": result['location'],
                    "caption": result['@search.captions'][0]['text'],
                    "index": index
                }
                if vector_search:
                    content[result['id']]["chunk"] = result['chunk']
                    content[result['id']]["score"] = result['@search.score']  # Uses the Hybrid RRF score

                else:
                    content[result['id']]["chunks"] = result['chunks']
                    content[result['id']]["score"] = result['@search.rerankerScore']  # Uses the reranker score
                    content[result['id']]["vectorized"] = result['vectorized']

    # After results have been filtered, sort and add the top k to the ordered_content
    if vector_search:
        topk = similarity_k
    else:
        topk = k * len(indexes)

    count = 0  # To keep track of the number of results added
    for site_id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[site_id] = content[site_id]
        count += 1
        if count >= topk:  # Stop after adding 5 results
            break

    return ordered_content


def update_vector_indexes(ordered_search_results: dict, embedder: OpenAIEmbeddings):
    """Get as input the results of a text-based multi-index search, vectorize the documents chunks that has not been done before and updates the vector-based indexes"""

    headers = {'Content-Type': 'application/json', 'api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    for key, value in ordered_search_results.items():
        if value["vectorized"] != True:  # If the document has not been vectorized yet
            i = 0
            for chunk in value["chunks"]:  # Iterate over the text chunks
                try:
                    upload_payload = {  # Insert the chunk and its vector/embedding in the vector-based index
                        "value": [
                            {
                                "id": key + "_" + str(i),
                                "title": f"{value['title']}_chunk_{str(i)}",
                                "chunk": chunk,
                                "chunkVector": embedder.embed_query(chunk if chunk != "" else "-------"),
                                "name": value["name"],
                                "location": value["location"],
                                "@search.action": "upload"
                            },
                        ]
                    }

                    r = requests.post(
                        os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + value["index"] + "-vector" + "/docs/index",
                        data=json.dumps(upload_payload), headers=headers, params=params)
                    if r.status_code != 200:
                        print(r.status_code)
                        print(r.text)
                    else:
                        i = i + 1  # increment chunk number

                except Exception as e:
                    print("Exception:", e)
                    print(chunk)
                    continue

        # Update document in text-based index and mark it as "vectorized"
        upload_payload = {
            "value": [
                {
                    "id": key,
                    "vectorized": True,
                    "@search.action": "merge"
                },
            ]
        }

        r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + value["index"] + "/docs/index",
                          data=json.dumps(upload_payload), headers=headers, params=params)


def get_answer(llm: AzureChatOpenAI,
               docs: List[Document],
               query: str,
               language: str,
               chain_type: str,
               memory: ConversationBufferMemory = None,
               callback_manager: BaseCallbackManager = None
               ) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    if chain_type == "stuff":
        if memory == None:
            chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                               prompt=COMBINE_PROMPT,
                                               callback_manager=callback_manager)
        else:
            chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                               prompt=COMBINE_CHAT_PROMPT,
                                               memory=memory,
                                               callback_manager=callback_manager)

    elif chain_type == "map_reduce":
        if memory == None:
            chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                               question_prompt=COMBINE_QUESTION_PROMPT,
                                               combine_prompt=COMBINE_PROMPT,
                                               callback_manager=callback_manager)
        else:
            chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                               question_prompt=COMBINE_QUESTION_PROMPT,
                                               combine_prompt=COMBINE_CHAT_PROMPT,
                                               memory=memory,
                                               callback_manager=callback_manager)
    else:
        print("Error: chain_type", chain_type, "not supported")

    answer = chain({"input_documents": docs, "question": query, "language": language}, return_only_outputs=True)

    return answer


def get_answer_customized(llm: AzureChatOpenAI,
                          docs: List[Document],
                          query: str,
                          language: str,
                          model: str,
                          completion_tokens: int = 1000,
                          memory: ConversationBufferMemory = None,
                          callback_manager: BaseCallbackManager = None) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""
    answer_started = time.time()

    # Test if the stuff method exceeds token limits
    chain_type = "stuff"
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                       prompt=COMBINE_CHAT_PROMPT,
                                       memory=memory,
                                       callback_manager=callback_manager)

    other_keys = {"question": query, "language": language}
    other_keys.update(memory.load_memory_variables({}))
    prompt_token = chain.prompt_length(docs, **other_keys)
    tokens_limit = model_tokens_limit(model)

    payload = {"input_documents": docs, "question": query, "language": language}
    if (prompt_token + completion_tokens) > 0.9 * tokens_limit:
        chain_type = "map_reduce"
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type,
                                           question_prompt=COMBINE_QUESTION_PROMPT,
                                           combine_prompt=COMBINE_CHAT_PROMPT,
                                           memory=memory,
                                           callback_manager=callback_manager)
        payload.update({"token_max": 16000})

    print(f"--- {chain_type} ---")

    # Log number of tokens used for documents, memory, prompt, and completion tokens
    answer = chain(inputs=payload, return_only_outputs=True)

    answer_ended = time.time()
    duration = answer_ended - answer_started
    doc_token = num_tokens_from_docs(docs)
    memory_token = num_tokens_from_string(memory.buffer_as_str)
    token_count = {
        "duration": duration,
        "start_time": answer_started,
        "end_time": answer_ended,
        "num_docs_tokens": doc_token,
        "num_memory_tokens": memory_token,
        "num_base_prompt_tokens": prompt_token - (doc_token + memory_token),
        "num_combined_prompt_tokens": prompt_token,
        "num_completion_tokens": num_tokens_from_string(answer["output_text"])
    }
    logger.debug(f"Token count: {token_count}", extra=token_count)

    log_bot_answer = {
        "user_input": query,
        "bot_output": answer["output_text"],
        "duration": duration,
        "start_time": answer_started,
        "end_time": answer_ended
    }
    logger.debug(f"Bot Answer: {log_bot_answer}", extra=log_bot_answer)

    return answer


def run_agent(question: str, agent_chain: AgentExecutor) -> str:
    """Function to run the brain agent and deal with potential parsing errors"""

    for i in range(5):
        try:
            response = agent_chain.run(input=question)
            break
        except OutputParserException as e:
            # If the agent has a parsing error, we use OpenAI model again to reformat the error and give a good answer
            chatgpt_chain = LLMChain(
                llm=agent_chain.agent.llm_chain.llm,
                prompt=PromptTemplate(input_variables=["error"],
                                      template='Remove any json formating from the below text, also remove any portion that says someting similar this "Could not parse LLM output: ". Reformat your response in beautiful Markdown. Just give me the reformated text, nothing else.\n Text: {error}'),
                verbose=False
            )

            response = chatgpt_chain.run(str(e))
            continue

    return response

def get_user_site_list(user_aad_id: str) -> List[dict]:
    payload = {
        "userId": user_aad_id
    }
    url = f"{os.environ['SHAREPOINT_HANDLER_URL']}/api/sharepoint/list-user-site"
    try:
        resp = requests.get(url, json=payload)
        resp.raise_for_status()
        site_list = resp.json()["Value"]
        return site_list
    except Exception as e:
        raise e


def extract_site_list_id(site_list: List[dict]) -> list:
    site_ids: List[Any] = []
    for site in site_list:
        site_ids.append(site["id"])
    return site_ids
