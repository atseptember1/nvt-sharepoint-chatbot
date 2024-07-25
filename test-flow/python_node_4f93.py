
import os
import random
import json
from typing import List

from promptflow.core import tool
from dotenv import load_dotenv
from langchain.memory import CosmosDBChatMessageHistory, ConversationBufferWindowMemory
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain_core.documents import Document

from utils import get_answer_customized


load_dotenv()
# Env variables needed by langchain
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
aoai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
tenant_id = os.environ.get("AZURE_TENANT_ID")
completion_tokens = int(os.environ.get("AZURE_OPENAI_COMPLETION_TOKEN"))
answer_language = os.environ.get("ANSWER_LANGUAGE")

# Aoai configuration
llm_list: list[AzureChatOpenAI] = []
aoai_resources: list = json.loads(os.environ.get("AUZRE_OPENAI_RESOURCES"))
if aoai_resources is not None:
    for resource in aoai_resources:
        if resource["model_name"].startswith("gpt"):
            llm_list.append(AzureChatOpenAI(
                openai_api_type="azure",
                openai_api_base=resource["endpoint"],
                openai_api_key=resource["key"],
                openai_api_version=aoai_api_version,
                deployment_name=resource["deployment_name"],
                temperature=0.5,
                max_tokens=completion_tokens
            ))
llms = llm_list[0].with_fallbacks(llm_list[1:])

# set random session id for each request
session_id = str(random.randint(0, 1000000000))
# user id is fixed for each request to simulate the same user
user_id = "00001"


@tool
def execute_chain(user_input: str, docs: List[Document]) -> str:
    memory = setup_cosmos()
    answer = get_answer_customized(llms, docs, user_input,
                                   answer_language, "gpt-35-turbo-16k", completion_tokens, memory, None)
    return answer["output_text"]


def setup_cosmos() -> ConversationBufferWindowMemory:
    cosmos = CosmosDBChatMessageHistory(
        cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
        cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
        cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
        connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
        session_id=session_id,
        user_id=user_id
    )
    cosmos.prepare_cosmos()
    return ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", return_messages=True,
                                          k=3, chat_memory=cosmos)
