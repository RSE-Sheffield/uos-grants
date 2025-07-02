#%%
import os
import asyncio
import random
import logging
from typing import List, Dict, Any

from neo4j import GraphDatabase
from langchain.embeddings import init_embeddings
from langchain_neo4j import Neo4jGraph


logger = logging.getLogger("uos_scheduler")

class GraphEmbedder:
    def __init__(
        self,
        embedding_model_name: str,
        embedding_model_provider: str,
        embedding_model_api_key: str,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
    ):
        logger.info("🔗 Initializing GraphEmbedder...")
        self.driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_username, neo4j_password)
        )
        self.graph = Neo4jGraph(
            url=neo4j_uri, username=neo4j_username, password=neo4j_password
        )
        self.embedding_model = init_embeddings(
            model=embedding_model_name,
            provider=embedding_model_provider,
            api_key=embedding_model_api_key,
        )
        logger.info("✅ GraphEmbedder initialized successfully.")

    def fetch_nodes_without_embedding(self, node_label: str) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Fetching nodes without embeddings for label: {node_label}")
        query = f"""
        MATCH (n:{node_label})
        WHERE n.embedding IS NULL
        RETURN n, elementId(n) AS node_id
        """
        results = self.graph.query(query)
        return [{**record["n"], "node_id": record["node_id"]} for record in results]

    @staticmethod
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i: i + chunk_size]

    async def embed_and_store_single(
        self, node: Dict[str, Any], node_key: str, max_retries: int = 8, base_delay: int = 10
    ):
        retries = 0
        while retries < max_retries:
            try:
                embedding = await self.embedding_model.aembed_query(node[node_key])
                self.graph.query(
                    """
                    MATCH (n)
                    WHERE elementId(n) = $node_id
                    SET n.embedding = $embedding
                    """,
                    params={
                        "node_id": node["node_id"],
                        "embedding": embedding,
                    },
                )
                logger.info(f"✅ Embedded {node[node_key]}")
                return
            except Exception as e:
                retries += 1
                delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, base_delay)
                logger.warning(
                    f"⚠️ Error embedding {node[node_key]} (attempt {retries}): {e}. Retrying in {int(delay)} seconds..."
                )
                await asyncio.sleep(delay)

        logger.error(f"❌ Failed to embed {node[node_key]} after {max_retries} retries.")

    async def embed_nodes(
        self,
        nodes: List[Dict[str, Any]],
        node_key: str,
        chunk_size: int = 50,
    ):
        chunk_no = 0
        for chunk in self.chunk_list(nodes, chunk_size):
            chunk_no += 1
            logger.info(f"🚀 Processing chunk {chunk_no} with {len(chunk)} {node_key}(s)...")
            tasks = [
                self.embed_and_store_single(node, node_key) for node in chunk
            ]
            await asyncio.gather(*tasks)
            logger.info(f"✅ Finished processing chunk {chunk_no}.")

    async def process_node_labels(
        self,
        node_labels: List[str],
        node_key: str = "id",
        chunk_size: int = 100,
    ):
        for label in node_labels:
            nodes = self.fetch_nodes_without_embedding(label)

            if not nodes:
                logger.info(f"⏭️ No {label} nodes without embeddings. Skipping...")
                continue

            logger.info(f"🧠 Found {len(nodes)} {label} nodes to embed.")
            await self.embed_nodes(nodes, node_key, chunk_size)

    def close(self):
        logger.info("🔌 Closing Neo4j connection.")
        self.driver.close()

def get_embedding_node_labels() -> List[str]:
    nodes = os.getenv("EMBEDDING_NODES", "")
    return [label.strip() for label in nodes.split(",") if label.strip()]

async def run_graph_embedding():
    logger.info("🚀 Starting graph embedding...")

    embedder = GraphEmbedder(
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        embedding_model_provider=os.getenv("EMBEDDING_MODEL_PROVIDER"),
        embedding_model_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
    )

    try:
        node_labels = get_embedding_node_labels()

        if not node_labels:
            logger.warning("⚠️ No node labels found in EMBEDDING_NODES. Skipping embedding.")
            return

        logger.info(f"🔍 Node labels to embed: {node_labels}")
        await embedder.process_node_labels(node_labels)

        logger.info("✅ Graph embedding job completed successfully.")
    except Exception as e:
        logger.exception(f"❌ Graph embedding job failed: {e}")
    finally:
        embedder.close()
# %%
