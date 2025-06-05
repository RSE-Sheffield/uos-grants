# rag/tools.py

from collections import defaultdict
import os
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from langchain.embeddings.base import init_embeddings
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv(".env")

embedding_model = init_embeddings(
    f"{os.getenv('EMBEDDING_MODEL_PROVIDER')}:{os.getenv('EMBEDDING_MODEL_NAME')}"
)

# ===============================RAG Schema====================================


class ResearchInterestQuery(BaseModel):
    query_text: str = Field(
        description="The query to search for in the RAG system."
    )
    top_k: int = Field(
        default=10,
        description="The number of top results to return from the RAG system.",
    )


class PersonQuery(BaseModel):
    person_name: str = Field(
        description="The name/partial name of the person to query in the graph database."
    )


class PersonFullProfileQuery(BaseModel):
    person_name: str = Field(
        description="The name of the person to retrieve a full profile for."
    )


class DepartmentResearchInterestQuery(BaseModel):
    departments: list[str] = Field(
        description="List of department names to filter researchers by."
    )
    interests: list[str] = Field(
        description="List of research interests to filter researchers by."
    )
    top_k: int = Field(
        default=10,
        description="The number of top results to return from the RAG system.",
    )


# =============================RAG FUNCTIONS===================================


@tool(args_schema=ResearchInterestQuery)
def research_interests_query(query_text, top_k=5):
    """Query the research interests of people in the graph database.
    Use this tool when a query is only about research interests or related topics."""
    query_embedding = embedding_model.embed_query(query_text)
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    cypher = """
    CALL db.index.vector.queryNodes(
        'research_interest_index',
        $top_k,
        $query_embedding
    ) YIELD node AS ri, score

    MATCH (p:Person)-[:HAS_RESEARCH_INTEREST]->(ri)
    OPTIONAL MATCH (p)-[r]->(n)

    RETURN
      p.id AS name,
      ri.id AS matched_interest,
      score,
      collect(DISTINCT {
        rel: type(r),
        target: tail(labels(n))[0],
        value: CASE 
            WHEN type(r) IN ['HAS_URL', 'HAS_EMAIL'] THEN toLower(n.id)
            ELSE n.id
        END
      }) AS related_info
    ORDER BY score DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        result = session.run(
            cypher, query_embedding=query_embedding, top_k=top_k
        )
        results = [record.data() for record in result]

    bases = []

    for entry in results:
        base = f"{entry['name']} is a researcher interested in '{entry['matched_interest']}'.\n"

        grouped = defaultdict(list)
        for rel in entry["related_info"]:
            key = (rel["rel"], rel["target"])
            grouped[key].append(rel["value"])

        for (rel, target), values in grouped.items():
            joined = ", ".join(values)
            base += f"{target}(s): {joined}.\n"
        bases.append(base)

    return bases


@tool(args_schema=PersonQuery)
def get_people_by_name(person_name) -> str:
    """Query the graph database for people by their name."""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    cypher = """
    MATCH (p:Person)
    WHERE toLower(p.id) CONTAINS toLower($person_name)
    OPTIONAL MATCH (p)-[r]->(n)
    RETURN p, collect(DISTINCT {
        rel: type(r),
        target: tail(labels(n))[0],
        value: CASE
            WHEN type(r) IN ['HAS_URL', 'HAS_EMAIL'] THEN toLower(n.id)
            ELSE n.id
        END
    }) AS related_info
    """

    with driver.session() as session:
        results = session.run(cypher, person_name=person_name)
        people = [result.data() for result in results]
        if len(people) == 0:
            return f"No people found with name containing '{person_name}'."
        elif len(people) == 1:
            person = people[0]
            base = f"Found {person['p']['id']} with related info:\n"
            grouped = defaultdict(list)
            for rel in person["related_info"]:
                key = (rel["rel"], rel["target"])
                grouped[key].append(rel["value"])

            for (rel, target), values in grouped.items():
                joined = ", ".join(values)
                base += f"{target}(s): {joined}.\n"
            return base
        else:
            base = f"Found {len(people)} people with name containing '{person_name}':\n"
            for person in people:
                grouped = defaultdict(list)
                for rel in person["related_info"]:
                    key = (rel["rel"], rel["target"])
                    grouped[key].append(rel["value"])

                for (rel, target), values in grouped.items():
                    joined = ", ".join(values)
                    base += f"{target}(s): {joined}.\n"
            return base


@tool(args_schema=PersonFullProfileQuery)
def get_person_full_profile(person_name) -> str:
    """Retrieve a full profile of a person from the graph database.
    Use this tool when a specific person's full profile is requested."""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    cypher = """
    MATCH (p:Person)
    WHERE toLower(p.id) CONTAINS toLower($person_name)

    OPTIONAL MATCH (p)-[r]->(n)

    OPTIONAL MATCH (doc:Document)-[:MENTIONS]->(p)

    RETURN p, 
        collect(DISTINCT {
            rel: type(r),
            target: tail(labels(n))[0],
            value: CASE
                WHEN type(r) IN ['HAS_URL', 'HAS_EMAIL'] THEN toLower(n.id)
                ELSE n.id
            END
        }) AS related_info,
        collect(DISTINCT doc) AS mentioned_in_docs
    """

    with driver.session() as session:
        person = session.run(cypher, person_name=person_name)
        return person.data()[0]["mentioned_in_docs"][0]["text"]


@tool(args_schema=DepartmentResearchInterestQuery)
def get_researchers_by_departments_and_interests(
    departments: list[str], interests: list[str], top_k: int = 10
) -> list[str]:
    """Get researchers by department and vector-matched research interests.
    This tool uses two vector searches: one for departments and one for research interests.
    Use this tool when a query asks for researchers in specific departments or schools with specific research interests."""
    # Step 1: Embed all interest and department strings
    interest_embeddings = [embedding_model.embed_query(i) for i in interests]
    department_embeddings = [
        embedding_model.embed_query(d) for d in departments
    ]

    # Step 2: Cypher query using two vector searches
    cypher = """
    UNWIND $dept_embeddings AS dept_emb
    CALL db.index.vector.queryNodes('department_index', $topK, dept_emb)
    YIELD node AS dept, score AS dept_score

    UNWIND $interest_embeddings AS interest_emb
    CALL db.index.vector.queryNodes('research_interest_index', $topK, interest_emb)
    YIELD node AS ri, score AS interest_score

    MATCH (p:Person)-[:HAS_DEPARTMENT]->(dept)
    MATCH (p)-[:HAS_RESEARCH_INTEREST]->(ri)

    RETURN DISTINCT p.id AS name
    """

    # Step 3: Run the query
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        results = session.run(
            cypher,
            dept_embeddings=department_embeddings,
            interest_embeddings=interest_embeddings,
            topK=top_k,
        )
        return [record["name"] for record in results]
