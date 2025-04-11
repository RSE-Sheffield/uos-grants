# %%
from psycopg_pool import AsyncConnectionPool

async with AsyncConnectionPool(
    conninfo="postgresql://postgres:postgres@postgres:5432/uos_grants"
) as pool:
    print(pool)
# %%
from uos_grants.db.vector_db import VectorDB
from uos_grants.embedding import EmbeddingModel

embedding = EmbeddingModel(
    provider=self.valves.EMBEDDING_MODEL_PROVIDER,
    model_name=self.valves.EMBEDDING_MODEL_NAME,
    api_key=self.valves.LLM_API_KEY,
    dimensions=self.valves.EMBEDDING_DIMENSIONS,
)

vectordb = VectorDB(
    embeddings=self.embedding,
    collection_name=self.valves.VECTOR_DB_COLLECTION,
    connection=self.valves.VECTOR_DB_URI,
    use_jsonb=True,
    table_name=self.valves.VECTOR_DB_COLLECTION,
)
