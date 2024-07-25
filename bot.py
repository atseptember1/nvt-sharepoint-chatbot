# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import asyncio
import time
import logging
import json
import redis
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import CosmosDBChatMessageHistory
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain_openai import AzureOpenAIEmbeddings

# custom libraries that we will use later in the app
from prompts import *

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.core.teams import TeamsInfo
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes

from utils import *
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# Env variables needed by langchain
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
aoai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
tenant_id = os.environ.get("AZURE_TENANT_ID")
completion_tokens = int(os.environ.get("AZURE_OPENAI_COMPLETION_TOKEN"))
answer_language = os.environ.get("ANSWER_LANGUAGE")

# Env variables needed by the bot
enable_site_id = os.environ.get("ENABLE_SITE_ID")
logger.debug(f"ENABLE_SITE_ID is set to: {str(enable_site_id)}")
# Set enable_site_id to True if not set
if enable_site_id.lower() == "true":
    enable_site_id = True
elif enable_site_id is None or enable_site_id.lower() == "":
    logger.debug("ENABLE_SITE_ID is not set, defaulting to True")
    enable_site_id = True
elif enable_site_id.lower() == "false":
    enable_site_id = False

user_sharepoint_cache = SharePointCache(os.environ.get("SHAREPOINT_SITE_CACHE_PERIOD"), os.environ.get("REDIS_HOST"), os.environ.get("REDIS_PORT"))

# Callback hanlder used for the bot service to inform the client of the thought process before the final response
class BotServiceCallbackHandler(BaseCallbackHandler):
    """Callback handler to use in Bot Builder Application"""

    def __init__(self, turn_context: TurnContext) -> None:
        self.tc = turn_context

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"LLM Error: {error}\n"))

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"Tool: {serialized['name']}"))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        if "Action Input" in action.log:
            action = action.log.split("Action Input:")[1]
            asyncio.run(self.tc.send_activity(f"\u2611 Searching: {action} ..."))
            asyncio.run(self.tc.send_activity(Activity(type=ActivityTypes.typing)))


# Bot Class
class MyBot(ActivityHandler):

    def __init__(self):
        self.aoai_resources: list = json.loads(os.environ.get("AUZRE_OPENAI_RESOURCES"))
        self.llms_preconf: list[AzureChatOpenAI] = []
        self.embed_llms: list[AzureOpenAIEmbeddings] = []
        if self.aoai_resources is not None:
            for resource in self.aoai_resources:
                if resource["model_name"].startswith("gpt"):
                    self.llms_preconf.append(AzureChatOpenAI(
                        openai_api_type="azure",
                        azure_endpoint=resource["endpoint"],
                        openai_api_key=resource["key"],
                        openai_api_version=aoai_api_version,
                        deployment_name=resource["deployment_name"],
                        temperature=0.5,
                        max_tokens=completion_tokens
                    ))
                if resource["model_name"] == "embed":
                    self.embed_llms.append(AzureOpenAIEmbeddings(
                        openai_api_type="azure",
                        azure_endpoint=resource["endpoint"],
                        openai_api_key=resource["key"],
                        openai_api_version=aoai_api_version,
                        deployment=resource["deployment_name"],
                        chunk_size=1
                    ))
        self.llms_postconf: list[AzureChatOpenAI] = []

    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE)

    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
        turn_start_time = time.time()
        logger.info("--- TURN BEGINNS ---")

        # Get the user's tenant id and AAD id
        if not turn_context.activity.channel_data.get("tenant"):
            user_tenant_id = ""
        else:
            user_tenant_id = turn_context.activity.channel_data["tenant"]["id"]
        user_aad_id = turn_context.activity.from_property.aad_object_id
        if user_aad_id is None:
            user_aad_id = ""
        user_information = {
            "user_aad_id": user_aad_id,
            "user_id": turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id,
            "tenant_id": user_tenant_id
        }
        logger.info(f"User AAD information: {user_information}", extra=user_information)

        # Extract metadata from user input
        input_text_metadata = dict()
        try:
            input_text_metadata["local_timestamp"] = turn_context.activity.local_timestamp.strftime(
                "%I:%M:%S %p, %A, %B %d of %Y")
            input_text_metadata["local_timezone"] = turn_context.activity.local_timezone
            input_text_metadata["locale"] = turn_context.activity.locale
        except Exception as err:
            logger.error(err)

        # Check if user is in the same tenant
        if user_tenant_id != tenant_id and tenant_id is not None:
            logger.debug(
                f"User {user_information['user_id']} is in tenant {user_information['tenant_id']} not in {tenant_id}")
            await turn_context.send_activity("Sorry, I can only talk to people in the same tenant.")
        else:
            user_input_text = turn_context.activity.text
            user_input_text_with_metada = user_input_text + "\n\n metadata:\n" + str(input_text_metadata)
            logger.debug(f"User input text: {user_input_text_with_metada}",
                         extra={"input_text": user_input_text})

            # Set Callback Handler
            cb_handler = BotServiceCallbackHandler(turn_context)
            cb_manager = CallbackManager(handlers=[cb_handler])

            # Set LLM
            for l in self.llms_preconf:
                l.callback_manager = cb_manager
                self.llms_postconf.append(l)

            llm = self.llms_postconf[0].with_fallbacks(self.llms_postconf[1:])
            embedder = self.embed_llms[0]  # TODO: this should utilize fallback for AzureOpenAIEmbeddings

            # Set brain Agent with persisten memory in CosmosDB
            session_id = turn_context.activity.conversation.id

            logger.info(f"Session ID: {session_id}")

            cosmos = CosmosDBChatMessageHistory(
                cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
                cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
                cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
                connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
                session_id=session_id,
                user_id=user_information["user_id"]
            )
            cosmos.prepare_cosmos()
            memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", return_messages=True,
                                                    k=3, chat_memory=cosmos)
            # agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, system_message=CUSTOM_CHATBOT_PREFIX, human_message=CUSTOM_CHATBOT_SUFFIX)
            # agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, handle_parsing_errors=True)

            # get user Sharepoint site list
            vector_indexes = [os.environ["AZURE_SEARCH_INDEX"]]
            similarity_k = 3  # top results from multi-vector-index similarity search

            await turn_context.send_activity(Activity(type=ActivityTypes.typing))

            if (user_information["user_aad_id"] != "" and user_information != None) and enable_site_id:
                
                user_sharepoint_sites_from_cache = user_sharepoint_cache.get_key(user_information["user_aad_id"])
                site_list = None
                
                if user_sharepoint_sites_from_cache == None:
                    logger.info("[DEBUG CACHE] CACHE EXPIRED OR NOT EXISTS - Get user sites permission from API")
                    site_list = get_user_site_list(user_information["user_aad_id"])
                    user_sharepoint_cache.set_key(user_information["user_aad_id"], site_list)
                else:
                    logger.info("[DEBUG CACHE] CACHE HIT - Get user sites permission from CACHE")
                    site_list = json.loads(user_sharepoint_sites_from_cache)
                
                site_id_list = extract_site_list_id(site_list)
                logger.debug(f"User site list: {site_id_list}")
                logger.info("Searching documents in user's site list")
                ordered_results = get_search_results(turn_context.activity.text,
                                                     vector_indexes,
                                                     sas_token=os.environ.get("BLOB_SAS_TOKEN"),
                                                     vector_search=True,
                                                     similarity_k=similarity_k,
                                                     query_vector=embedder.embed_query(turn_context.activity.text),
                                                     site_ids=site_id_list,
                                                     enable_site_id=enable_site_id)
            else:
                ordered_results = get_search_results(turn_context.activity.text,
                                                     vector_indexes,
                                                     sas_token=os.environ.get("BLOB_SAS_TOKEN"),
                                                     vector_search=True,
                                                     similarity_k=similarity_k,
                                                     query_vector=embedder.embed_query(turn_context.activity.text),
                                                     enable_site_id=enable_site_id)

            top_docs = []
            for key, value in ordered_results.items():
                location = value["location"] if value["location"] is not None else ""
                top_docs.append(
                    Document(page_content=value["chunk"], metadata={"source": location + os.environ['BLOB_SAS_TOKEN']}))

            # send type acitivities so user awares that the message is being processed
            logger.info("Getting answer from OpenAI")
            answer = get_answer_customized(llm, top_docs, user_input_text_with_metada,
                                           answer_language, "gpt-35-turbo-16k", completion_tokens, memory, None)
            await turn_context.send_activity(answer["output_text"])

            # calculate turn duration
            logger.info("--- TURN ENDS ---")
            turn_end_time = time.time()
            turn_duration = turn_end_time - turn_start_time
            turn_duration_data = {
                "duration": turn_duration,
                "start_time": turn_start_time,
                "end_time": turn_end_time
            }
            logger.info(f"turn duration: {turn_duration_data}", extra=turn_duration_data)
