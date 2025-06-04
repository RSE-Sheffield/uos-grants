from pydantic import BaseModel, Field
from fastapi import Request
import json
import os
import logging

from uos_grants.rag.react_agent import get_react_agent
from psycopg_pool import AsyncConnectionPool

from langchain.chat_models.base import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_postgres import PGVector

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        CHAT_MEMORY_DB_URI: str = Field(
            default="",
            description="URI for the database connection for chat memory",
        )
        VECTOR_DB_URI: str = Field(
            default="",
            description="URI for the database connection for vector DB",
        )
        VECTOR_DB_COLLECTION: str = Field(
            default="",
            description="Collection name for the vector DB",
        )
        VECTOR_DB_TABLE_NAME: str = Field(
            default="",
            description="Table name for the vector DB",
        )
        EMBEDDING_MODEL_NAME: str = Field(
            default="",
            description="Name of the embedding model to use",
        )
        EMBEDDING_MODEL_PROVIDER: str = Field(
            default="",
            description="Provider for the embedding model",
        )
        EMBEDDING_DIMENSIONS: int = Field(
            default=3072,
            description="Dimensions of the embedding model",
        )
        LLM_MODEL_PROVIDER: str = Field(
            default="",
            description="Provider for the LLM model",
        )
        LLM_MODEL: str = Field(
            default="",
            description="LLM model to use",
        )
        LLM_API_KEY: str = Field(
            default="",
            description="API key for the LLM",
        )

    def __init__(self):
        pass

    async def get_chat_id(self, request: Request):
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")
        chat_id = json.loads(body_str)["chat_id"]
        return chat_id

    async def run_graph_streaming(self, graph, body, config):
        async for event in graph.astream_events(body, config=config):
            if event["type"] == "stream":
                yield event["value"]

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __request__: Request = None,
    ) -> str:
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        self.valves.VECTOR_DB_COLLECTION = self.valves.VECTOR_DB_COLLECTION
        self.errors = []
        self.embedding = init_embeddings(
            model=self.valves.EMBEDDING_MODEL_NAME,
            provider=self.valves.EMBEDDING_MODEL_PROVIDER,
            api_key=self.valves.LLM_API_KEY,
        )
        self.connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        self.vectordb = PGVector(
            embeddings=self.embedding,
            collection_name=self.valves.VECTOR_DB_COLLECTION,
            connection=self.valves.VECTOR_DB_URI,
            use_jsonb=True,
            # async_mode=True,
        )
        self.llm = init_chat_model(
            model=self.valves.LLM_MODEL,
            model_provider=self.valves.LLM_MODEL_PROVIDER,
            api_key=self.valves.LLM_API_KEY,
        )
        self.retriever = self.vectordb
        # Use the unified endpoint with the updated signature
        # self.chat_id = await self.get_chat_id(__request__)
        self.connection_pool = AsyncConnectionPool(
            conninfo=self.valves.CHAT_MEMORY_DB_URI, kwargs=self.connection_kwargs
        )
        async with self.connection_pool as pool:
            self.memory = AsyncPostgresSaver(conn=pool)
            try:
                await self.memory.setup()
            except Exception as e:
                logger.error(f"Error setting up memory: {e}")
                pass
            self.chat_id = await self.get_chat_id(__request__)
            config = {"configurable": {"thread_id": self.chat_id}}
            graph = get_react_agent(self.llm, memory=self.memory)
            print("loaded up ot graph")

            async for event in graph.astream_events(body, config=config):
                if event["event"] == "on_chat_model_stream":
                    yield (str(event["data"]["chunk"].content))

            # async for event in graph.astream_events(body, config=config):
            #     # Only forward chunks that are actual stream events
            #     if "langgraph_node" in event["metadata"]:
            #         if event["metadata"]["langgraph_node"] == "generate_response":
            #             if event["event"] == "on_chat_model_stream":
            #                 yield str(event["data"]["chunk"].content) + ""
            # if event["event"] == "on_chat_model_stream":
            #    yield str(event["data"]["chunk"].content)
            # if "on_chat_model_stream" in event:
            # yield event["data"]["chunk"].content
            # if "stream" in event:
            #     yield {"type": "stream", "value": event["stream"]}
