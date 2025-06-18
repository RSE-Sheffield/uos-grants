# %%
from neo4j import GraphDatabase
import os

from langchain.embeddings import init_embeddings
from langchain_neo4j import Neo4jGraph
import asyncio

# from dotenv import load_dotenv

# load_dotenv(".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)


embedding_model = init_embeddings(
    model=os.getenv("EMBEDDING_MODEL_NAME"),
    provider=os.getenv("EMBEDDING_MODEL_PROVIDER"),
    api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),  # noqa: E501
)

graph = Neo4jGraph()


def chunk_list(lst, chunk_size):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


# Retry wrapper for embedding + store
async def embed_and_store_single(result, node_key, embedding_model, graph, retry_delay=60):
    """Embed a single interest and store it in the graph database."""
    while True:
        try:
            embedding = await embedding_model.aembed_query(result[node_key])
            graph.query(
                """
                MATCH (r)
                WHERE elementId(r) = $node_id
                SET r.embedding = $embedding
                """,
                params={
                    "node_id": result["node_id"],
                    "embedding": embedding,
                },
            )
            return  # success
        except Exception as e:
            print(f"Error processing interest {result[node_key]}: {e}")
            await asyncio.sleep(retry_delay)  # back off and retry


# Run batches concurrently
async def embed_and_store_node(results, node_key, chunk_size=50):
    """Embed and store node in chunks."""
    chunk_no = 0
    for chunk in chunk_list(results, chunk_size):
        chunk_no += 1
        print(f"Processing chunk {chunk_no} with {len(chunk)} {node_key}(s)...")
        tasks = [
            embed_and_store_single(result, node_key, embedding_model, graph) for result in chunk
        ]
        await asyncio.gather(*tasks)
        print(f"Finished processing chunk {chunk_no}.")
