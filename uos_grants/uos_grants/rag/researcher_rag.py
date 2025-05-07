# uos_grants/rag/researcher_rag.py

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import logging
import os

from typing import List, Union, Generator, Iterator

from pydantic import BaseModel, Field

from langchain.chat_models.base import init_chat_model
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END
from functools import partial
from langgraph.checkpoint.memory import MemorySaver
import uuid

from uos_grants.db.vector_db import VectorDB
from uos_grants.embedding import EmbeddingModel

from psycopg_pool import ConnectionPool

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
        self.name = "sheffield Researcher RAG"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model__fields__.items()}
        )
        self.embedding = EmbeddingModel(
            provider=self.valves.EMBEDDING_MODEL_PROVIDER,
            model_name=self.valves.EMBEDDING_MODEL_NAME,
            api_key=self.valves.LLM_API_KEY,
            dimensions=self.valves.EMBEDDING_DIMENSIONS,
        )
        self.vectordb = VectorDB(
            embeddings=self.embedding,
            collection_name=self.valves.VECTOR_DB_COLLECTION,
            connection=self.valves.VECTOR_DB_URI,
            use_jsonb=True,
            table_name=self.valves.VECTOR_DB_TABLE_NAME,
        )
        
        self.retriever = await self.vectordb.aget_vectorstore(self.valves.VECTOR_DB_COLLECTION)
        self.connection_pool = ConnectionPool(conninfo=self.valves.CHAT_MEMORY_DB_URI)
        self.llm = init_chat_model(model=self.valves.LLM_MODEL, model_provider=self.valves.LLM_MODEL_PROVIDER, configurable_fields={'api_key': self.valves.LLM_API_KEY})
        self.memory = PostgresSaver()

        self.custom_retriever_tool = Tool(
            name="CustomRetriever",
            func=self.custom_retrieve,
            description="Retrieves and formats documents with metadata and content.",
        )

    async def on_startup(self):
        """
        Initialize the pipeline.
        """
        pass

    async def on_shutdown(self):
        """
        Shutdown the pipeline.
        """
        pass
    
    async def pipe(self, body: dict):
        print(body)