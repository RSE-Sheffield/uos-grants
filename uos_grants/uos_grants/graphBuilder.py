# graph_sync.py

import asyncio
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict

from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import create_unstructured_prompt
from langchain_neo4j import Neo4jGraph

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from sqlalchemy import select
from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import Researcher

from openai import RateLimitError
from tqdm import tqdm


# === Disable LangSmith Tracing Globally ===
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# === Logging Setup ===
log_dir = Path("/app/logs/graph_db")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"graph_sync_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("graph_sync")

# === Suppress LangSmith Multipart Errors ===
class LangSmithFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not (
            "LangSmithRateLimitError" in msg
            or "multipart ingest runs" in msg
            or "api.smith.langchain.com" in msg
        )

logging.getLogger().addFilter(LangSmithFilter())
logging.getLogger("langchain").addFilter(LangSmithFilter())
logging.getLogger("openai").addFilter(LangSmithFilter())


class GraphSync:
    def __init__(self):
        logger.info("GraphSync initialized. Neo4j connection will be opened when sync starts.")

        # === LLM Setup ===
        self.llm = init_chat_model(
            model=os.getenv("GRAPH_LLM_MODEL"),
            model_provider=os.getenv("GRAPH_LLM_PROVIDER"),
            api_key=os.getenv("GRAPH_LLM_API_KEY"),
        )

        self.prompt = create_unstructured_prompt(
            node_labels=["Research_Interest"],
            rel_types=[("Person", "has_research_interest", "Research_Interest")],
            relationship_type="tuple",
            additional_instructions=(
                "This document describes the research interests of a person. "
                "Extract the research interests as individual research interest nodes. "
                "Do NOT create a node for the person — the person is already represented in the graph with their name and url. "
                "Do NOT create a generic 'Research Interests' node. Only create nodes for specific research topics. "
                "Each research interest should be concise (maximum three words), with acronyms expanded (e.g., NLP → Natural Language Processing (NLP)). "
                "Do not hallucinate any information."
            ),
        )

        self.transformer = LLMGraphTransformer(llm=self.llm, prompt=self.prompt)

        # === Lazy-load Neo4j graph
        self.graph = None

    def connect_graph(self):
        if not self.graph:
            logger.info("Connecting to Neo4j graph...")
            self.graph = Neo4jGraph(url=os.getenv("NEO4J_URI", "bolt://neo4j:7687"))

    @staticmethod
    def safe_property(value):
        return value.strip() if value and isinstance(value, str) else None

    @staticmethod
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    @staticmethod
    def construct_full_profile(researcher: Researcher) -> str:
        fields = [
            f"Name: {researcher.name}" if researcher.name else None,
            f"Department: {researcher.department}" if researcher.department else None,
            f"URL: {researcher.url}",
            f"Main Role: {researcher.main_role}" if researcher.main_role else None,
            f"Additional Roles: {researcher.additional_roles}" if researcher.additional_roles else None,
            f"Email: {researcher.email}" if researcher.email else None,
            f"Telephone: {researcher.telephone}" if researcher.telephone else None,
            f"Address: {researcher.address}" if researcher.address else None,
            f"Profile: {researcher.profile}" if researcher.profile else None,
            f"Research Interests: {researcher.research_interests}" if researcher.research_interests else None,
        ]
        return "\n\n".join(filter(None, fields))

    async def process_research_interests_bulk(self, batch: List[Researcher]) -> Dict[str, List[Node]]:
        docs = []
        name_map = {}

        for researcher in batch:
            name = self.safe_property(researcher.name) or self.safe_property(researcher.url)
            if researcher.research_interests:
                content = (
                    f"This document describes the research interests of {name}. "
                    "Extract the specific research topics they work on as individual research interest nodes. "
                    f"The research interests are: {researcher.research_interests}"
                )
                doc = Document(page_content=content)
                docs.append(doc)
                name_map[doc.page_content] = name

        if not docs:
            return {}

        try:
            graph_documents = await self.transformer.aconvert_to_graph_documents(docs)
            logger.info(f"Processed {len(graph_documents)} research interest documents in batch.")

            result = {}
            for doc in graph_documents:
                nodes = []
                name = name_map.get(doc.source.page_content)

                for node in doc.nodes:
                    if (
                        node.type.lower() == "research_interest"
                        and node.id.lower() not in [name.lower(), "research interests"]
                        and node.id.strip() != ""
                    ):
                        nodes.append(
                            Node(
                                id=node.id.strip(),
                                type="Research_Interest",
                                properties={"id": node.id.strip()},
                            )
                        )
                result[name] = nodes

            return result

        except RateLimitError as e:
            logger.warning(f"Rate limit error on batch: {e}. Retrying after 60 seconds...")
            await asyncio.sleep(60)
            return await self.process_research_interests_bulk(batch)
        except Exception as e:
            logger.error(f"Error processing research interest batch: {e}")
            return {}

    def delete_person_and_relationships(self, name: str, url: str):
        logger.info(f"Deleting node and direct relationships for {name} ({url})")
        self.graph.query(
            """
            MATCH (p:Person {name: $name, url: $url})
            OPTIONAL MATCH (p)-[r]-()
            DELETE r, p
            """,
            params={"name": name, "url": url},
        )

    def delete_orphans(self, existing_urls: Set[str]):
        logger.info("Checking for orphaned people in graph...")
        result = self.graph.query("MATCH (p:Person) RETURN p.url AS url")
        neo4j_urls = {record["url"] for record in result}

        to_delete = neo4j_urls - existing_urls
        logger.info(f"Found {len(to_delete)} orphaned people to delete.")

        with tqdm(total=len(to_delete), desc="Deleting orphans") as pbar:
            for url in to_delete:
                logger.info(f"Deleting orphaned person {url}")
                self.graph.query(
                    """
                    MATCH (p:Person {url: $url})
                    OPTIONAL MATCH (p)-[r]-()
                    WITH p, collect(r) AS rels
                    FOREACH (r IN rels | DELETE r)
                    DELETE p
                    """,
                    params={"url": url},
                )
                self.graph.query(
                    """
                    MATCH (n)
                    WHERE NOT (n)--()
                    DELETE n
                    """
                )
                logger.info(f"Deleted orphaned person {url}.")
                pbar.update(1)

    async def sync_graph_from_db(self):
        self.connect_graph()

        logger.info("=== Starting graph sync job ===")
        async for db in get_session():
            result = await db.execute(select(Researcher))
            researchers = result.scalars().all()

            db_urls = {self.safe_property(r.url) for r in researchers if self.safe_property(r.url)}
            logger.info(f"Loaded {len(db_urls)} researchers from PostgreSQL.")

            BATCH_SIZE = 50

            with tqdm(total=len(researchers), desc="Processing researchers") as pbar:
                for batch in self.chunk_list(researchers, BATCH_SIZE):
                    logger.info(f"Processing batch of {len(batch)} researchers...")

                    researchers_to_update = []
                    researchers_skipped = []

                    for researcher in batch:
                        name = self.safe_property(researcher.name)
                        url = self.safe_property(researcher.url)
                        last_modified = self.safe_property(researcher.last_modified)

                        if not url:
                            logger.warning(f"Skipping researcher with missing URL: {name}")
                            pbar.update(1)
                            continue

                        existing = self.graph.query(
                            """
                            MATCH (p:Person {name: $name, url: $url})
                            RETURN p.last_modified AS last_modified
                            """,
                            params={"name": name, "url": url},
                        )

                        if existing:
                            existing_last_modified = existing[0].get("last_modified")
                            if existing_last_modified != last_modified:
                                logger.info(f"{url} has changed. Updating node.")
                                self.delete_person_and_relationships(name, url)
                                researchers_to_update.append(researcher)
                            else:
                                logger.info(f"{url} is up to date. Skipping update.")
                                researchers_skipped.append(researcher)
                                pbar.update(1)
                        else:
                            logger.info(f"{url} is new in graph.")
                            researchers_to_update.append(researcher)

                    if researchers_to_update:
                        research_interest_results = await self.process_research_interests_bulk(
                            researchers_to_update
                        )
                    else:
                        research_interest_results = {}

                    for researcher in researchers_to_update:
                        name = self.safe_property(researcher.name)
                        url = self.safe_property(researcher.url)
                        last_modified = self.safe_property(researcher.last_modified)
                        timestamp = datetime.utcnow().isoformat()

                        person_node = Node(
                            id=name,
                            type="Person",
                            properties={
                                "name": name,
                                "url": url,
                                "last_modified": last_modified,
                                "add_time": timestamp,
                            },
                        )

                        full_profile_text = self.construct_full_profile(researcher)

                        full_profile_node = Node(
                            id=f"{name}_profile",
                            type="Full_Profile",
                            properties={
                                "content": full_profile_text,
                                "add_time": timestamp,
                            },
                        )

                        nodes = [person_node, full_profile_node]
                        relationships = [
                            Relationship(
                                source=person_node,
                                target=full_profile_node,
                                type="has_full_profile",
                                properties={"add_time": timestamp},
                            )
                        ]

                        def maybe_add_node(label, rel_type, value):
                            if value:
                                node = Node(
                                    id=value,
                                    type=label,
                                    properties={
                                        "id": value,
                                        "add_time": timestamp,
                                    },
                                )
                                nodes.append(node)
                                relationships.append(
                                    Relationship(
                                        source=person_node,
                                        target=node,
                                        type=rel_type,
                                        properties={"add_time": timestamp},
                                    )
                                )

                        maybe_add_node("Department", "has_department", self.safe_property(researcher.department))
                        maybe_add_node("Email", "has_email", self.safe_property(researcher.email))
                        maybe_add_node("Telephone", "has_telephone", self.safe_property(researcher.telephone))
                        maybe_add_node("Address", "has_address", self.safe_property(researcher.address))
                        maybe_add_node("Role", "has_role", self.safe_property(researcher.main_role))

                        if researcher.additional_roles:
                            roles = [
                                r.strip() for r in researcher.additional_roles.split(",") if r.strip()
                            ]
                            for role in roles:
                                maybe_add_node("Role", "has_role", role)

                        research_interest_nodes = research_interest_results.get(name, [])
                        for ri_node in research_interest_nodes:
                            ri_node.properties["add_time"] = timestamp
                            nodes.append(ri_node)
                            relationships.append(
                                Relationship(
                                    source=person_node,
                                    target=ri_node,
                                    type="has_research_interest",
                                    properties={"add_time": timestamp},
                                )
                            )

                        graph_doc = GraphDocument(
                            nodes=nodes,
                            relationships=relationships,
                            source=Document(page_content=f"Database record for {name} ({url})"),
                        )

                        self.graph.add_graph_documents([graph_doc])

                        logger.info(
                            f"Uploaded {name} with {len(nodes)} nodes and {len(relationships)} relationships."
                        )
                        pbar.update(1)

                    for researcher in researchers_skipped:
                        logger.info(
                            f"Skipped {researcher.name} ({researcher.url}) - No changes."
                        )

            self.delete_orphans(db_urls)

        logger.info("=== Graph sync job completed ===")

async def sync_graph_from_db():
    graph_sync = GraphSync()
    await graph_sync.sync_graph_from_db()
    logger.info("Graph sync from database completed successfully.")