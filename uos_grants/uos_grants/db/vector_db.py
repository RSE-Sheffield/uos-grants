import os
from uos_grants.embedding import EmbeddingModel

from langchain_postgres import PGVector
from langchain_postgres import PGEngine

# PROVIDER = os.getenv("PROVIDER", "openai")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# VECTOR_DB_URL = os.getenv(
#     "VECTOR_DB_URL",
#     "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
# )
# COLLECTION_NAME = os.getenv("CHAT_HISTORY_COLLECTION_NAME")

# if COLLECTION_NAME is None:
#     raise ValueError("CHAT_HISTORY_COLLECTION_NAME must be set in environment variables.")

# embedding = EmbeddingModel(
#     provider=PROVIDER,
#     model_name=EMBEDDING_MODEL_NAME,
#     api_key=OPENAI_API_KEY,
# )


class VectorDB:
    """
    Class to manage the vector database.

    Attributes:
        embeddings (EmbeddingModel): The embedding model instance.
        collection_name (str): The name of the collection in the vector database.
        connection (str): The connection string for the vector database.
        use_jsonb (bool): Whether to use JSONB for storing vectors.
    """

    def __init__(
        self,
        embeddings: EmbeddingModel,
        collection_name: str,
        connection: str,
        use_jsonb: bool,
        table_name: str = None,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.connection = connection
        self.use_jsonb = use_jsonb
        self.engine = PGEngine.from_connection_string(url=connection)
        self.vector_size = self.embeddings.dimensions or 3072
        self.table_name = table_name

    async def ainit(self):
        await self.engine.ainit_vectorstore_table(
            table_name=self.table_name,
            vector_size=self.vector_size,
        )

    async def aget_vectorstore(self, table_name: str):
        try:
            vectorstore = await PGVectorStore.create(
                engine=self.engine,
                table_name=table_name,
                embedding_service=self.embeddings,
            )
            return vectorstore
        except ValueError:
            await self.ainit()
            vectorstore = await PGVectorStore(
                engine=self.engine,
                table_name=table_name,
                embedding_service=self.embeddings,
                use_jsonb=True,
            )
            return vectorstore
    
    def get_vectorstore(self, table_name: str):
        vectorstore = PGVector(
            engine=self.engine,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            use_jsonb=True,
        )
        return vectorstore

    async def aadd_documents(self, documents: list, table_name: str = None):
        """
        Add documents to the vector database.

        Args:
            documents (list): List of documents to add.
            metadata (dict): Metadata to associate with the documents.
        """
        if table_name is None:
            table_name = self.table_name

        vectorstore = await self.aget_vectorstore(table_name)
        if not isinstance(documents, list):
            documents = [documents]
        store = await vectorstore.aadd_documents(documents=documents)

        return store
    
    def add_documents(self, documents: list, table_name: str = None):
        """
        Add documents to the vector database.

        Args:
            documents (list): List of documents to add.
            metadata (dict): Metadata to associate with the documents.
        """
        if table_name is None:
            table_name = self.table_name

        vectorstore = self.get_vectorstore(table_name)
        if not isinstance(documents, list):
            documents = [documents]
        store = vectorstore.add_documents(documents=documents)

        return store

    async def search_documents(
        self, query: str, k: int = 5, filter: dict = None, table_name: str = None
    ) -> list:
        """
        Search for documents in the vector database.

        Args:
            query (str): The query string to search for.
            k (int): The number of results to return.
            filter (dict): Optional filter for the search.

        Returns:
            list: List of documents matching the query.
        """
        if table_name is None:
            table_name = self.table_name
        vectorstore = await self.get_vectorstore(table_name)
        results = await vectorstore.asimilarity_search(query=query, k=k, filter=filter)
        return results