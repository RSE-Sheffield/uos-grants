# %%
# graph_generator.py

from langchain.chat_models.base import init_chat_model
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jGraph

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import (
    create_unstructured_prompt,
)

from node_embedding import embed_and_store_node

from openai import RateLimitError
import os

import glob

import asyncio

llm = init_chat_model(
    model=os.getenv("GRAPH_LLM_MODEL"),
    provider=os.getenv("GRAPH_LLM_PROVIDER"),
    api_key=os.getenv("GRAPH_LLM_API_KEY"),
)

researchers = glob.glob(f"{os.getenv('RESEARCHER_TXT_PATH')}/*.txt")

doc_texts = [Path(researcher).read_text() for researcher in researchers]
documents = [Document(page_content=doc_text) for doc_text in doc_texts]

node_labels = [
    "Person",
    "Department",
    "Email",
    "Telephone",
    "Address",
    "URL",
    "Role",
    "Research_Interest",
    "External_Affiliation",
    "Research_Group",
]

rel_types = [
    ("Person", "has_department", "Department"),
    ("Person", "has_email", "Email"),
    ("Person", "has_telephone", "Telephone"),
    ("Person", "has_address", "Address"),
    ("Person", "has_url", "URL"),
    ("Person", "has_role", "Role"),
    ("Person", "has_research_interest", "Research_Interest"),
    ("Person", "has_external_affiliation", "External_Affiliation"),
    ("Person", "has_research_group", "Research_Group"),
]

instructions = (
    "This is a university staff profile. Identify the person, and then extract their attributes. "
    "Focus especially on capturing research interests and contact information."
    "If research interests are combined, seperate them into their own research interest nodes."
    "Research interests should be concise and no more than 3 words long. If there are multiple research interests, create separate nodes for each."
    "Research interests that use acronyms should include an expanded version in parentheses."
    "If the person has multiple roles, create separate nodes for each role."
    "If the person has multiple departments, create separate nodes for each department."
    "If the person has multiple email addresses, create separate nodes for each email address."
    "If the person has multiple telephone numbers, create separate nodes for each telephone number."
    "If the person has multiple addresses, create separate nodes for each address."
    "If the person has multiple URLs, create separate nodes for each URL."
    "Attempt to extract as much information as possible, but do not make up any information."
    "If a person has affiliations with universities or institutions outside of the main university, create a separate node for each external affiliation."
    "If a person has multiple external affiliations, create separate nodes for each external affiliation."
    "Email addresses should be all lowercase and should not contain any spaces or special characters other than @ and ."
    "URLs should be in the format of a valid URL, starting with http:// or https://. and should be lowercase."
)

prompt = create_unstructured_prompt(
    node_labels=node_labels,
    rel_types=rel_types,
    relationship_type="tuple",
    additional_instructions=instructions,
)

# Run the graph transformer
transformer = LLMGraphTransformer(llm=llm, prompt=prompt)


# Helper function to chunk a list
def chunk_list(lst: List, chunk_size: int) -> List[List]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


# Your async processing function
async def process_documents_in_chunks(transformer, documents, chunk_size=50):
    graph = Neo4jGraph()
    print("Processing documents in chunks...")
    print(
        f"Total chunks: {len(documents) // chunk_size + (1 if len(documents) % chunk_size > 0 else 0)}"
    )

    chunk_count = 0

    for chunk in chunk_list(documents, chunk_size):
        chunk_count += 1
        print(f"\nProcessing chunk {chunk_count}...")
        print(f"Chunk size: {len(chunk)}")

        while True:
            try:
                results = await transformer.aconvert_to_graph_documents(chunk)
                graph.add_graph_documents(
                    results, baseEntityLabel=True, include_source=True
                )
                break  # Success! Exit the retry loop.
            except RateLimitError as e:
                print(f"Rate limit error: {e}. Retrying after 60 seconds...")
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying after 60 seconds...")
                await asyncio.sleep(60)


# Usage
if __name__ == "__main__":
    # Run the async function
    graph = Neo4jGraph()

    asyncio.run(process_documents_in_chunks(transformer, documents))

    research_interests = graph.query("MATCH (r:Research_interest) RETURN elementId(r) AS node_id, r.id AS interest")
    asyncio.run(embed_and_store_node(research_interests, "interest"))

    departments = graph.query("MATCH (d:Department) RETURN elementId(d) AS node_id, d.id AS department")
    asyncio.run(embed_and_store_node(departments, "department", chunk_size=1000))

    roles = graph.query("MATCH (r:Role) RETURN elementId(r) AS node_id, r.id AS role")
    asyncio.run(embed_and_store_node(roles, "role", chunk_size=1000))


# %%
