# rag/tools.py

from collections import defaultdict
import os
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

from langchain.chat_models.base import init_chat_model
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
        min_length=1, description="The query to search for in the RAG system."
    )
    top_k: int = Field(
        default=10,
        description="The number of top results to return from the RAG system.",
    )


class PersonQuery(BaseModel):
    person_name: str = Field(
        min_length=1,
        description="The name/partial name of the person to query in the graph database.",
    )


class PersonFullProfileQuery(BaseModel):
    person_name: str = Field(
        min_length=1,
        description="The name of the person to retrieve a full profile for.",
    )


class DepartmentResearchInterestQuery(BaseModel):
    departments: list[str] = Field(
        min_items=1,
        description="List of department names to filter researchers by.",
    )
    interests: list[str] = Field(
        min_items=1,
        description="List of research interests to filter researchers by.",
    )
    top_k: int = Field(
        default=10,
        description="The number of top results to return from the RAG system.",
    )


# =============================RAG FUNCTIONS===================================


@tool
def scrape_url_content(url: str) -> str:
    """
    Scrape the main content from a webpage URL.

    ✅ Use this tool to retrieve the full text content from a grant call,
    researcher profile, or other public webpage.

    It tries to extract from <div id="block-uos-public-content"> first,
    then falls back to <main> or <body> if not present.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"❌ Failed to fetch page: HTTP {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        content = (
            soup.find("div", id="block-uos-public-content")
            or soup.find("main")
            or soup.find("body")
        )

        if not content:
            return "❌ Failed to extract meaningful content from the page."

        text = content.get_text(separator="\n", strip=True)
        return text

    except Exception as e:
        return f"❌ Error fetching page: {str(e)}"


@tool
def extract_grant_topics(text: str) -> list[str]:
    """
    Extract key research topics or interests from a grant call or descriptive text.

    ✅ Use this tool after scraping a grant page to get relevant research areas.

    Output is a list of concise research topics.
    """
    llm = init_chat_model(
        model=os.getenv("GRAPH_LLM_MODEL"),
        model_provider=os.getenv("GRAPH_LLM_PROVIDER"),
        api_key=os.getenv("GRAPH_LLM_API_KEY"),
    )

    prompt = f"""
    You are an assistant that extracts structured information from research grant descriptions.

    Given the following text:

    \"\"\"{text}\"\"\"

    List the key research topics, fields of study, or research interests relevant to this grant.
    Provide a comma-separated list, avoiding overly generic terms like "research" or "science".

    Output ONLY the list.
    """

    try:
        response = llm.invoke(prompt)
        interests = [
            i.strip() for i in response.content.split(",") if i.strip()
        ]
        if not interests:
            return ["⚠️ No research interests extracted."]
        return interests
    except Exception as e:
        return [f"❌ Failed to extract interests: {str(e)}"]


@tool
def match_departments_to_interests(interests: list[str]) -> list[str]:
    """
    Match research interests to known departments in the database.

    ✅ Use this tool to map extracted grant topics to departments.

    Example: "Artificial Intelligence" → "School of Computer Science"
    """
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        dept_results = session.run(
            """
            MATCH (d:Department)
            WHERE ANY(interest IN $interests WHERE toLower(d.id) CONTAINS toLower(interest))
            RETURN d.id AS department
            """,
            interests=interests,
        )
        departments = [record["department"] for record in dept_results]

    if not departments:
        return ["⚠️ No departments matched the given interests."]
    return departments


@tool(args_schema=ResearchInterestQuery)
def research_interests_query(query_text, top_k=5):
    """Query the research interests of people in the graph database.
    Use this tool when a query is only about research interests or related topics.
    """

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
    DISTINCT p.name AS name,
    p.url AS url,
    ri.id AS matched_interest,
    score,
    collect(DISTINCT {
        rel: type(r),
        target: head(labels(n)),
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
        base = f"{entry['name']} ({entry['url']}) is a researcher interested in '{entry['matched_interest']}'.\n"

        grouped = defaultdict(list)
        for rel in entry["related_info"]:
            key = (rel["rel"], rel["target"])
            grouped[key].append(rel["value"])

        for (rel, target), values in grouped.items():
            joined = ", ".join(values)
            base += f"{target}(s): {joined}.\n"
        bases.append(base)

    return bases


# @tool(args_schema=PersonQuery)
# def get_people_by_name(person_name) -> str:
#     """Query the graph database for people by their name."""
#     driver = GraphDatabase.driver(
#         os.getenv("NEO4J_URI"),
#         auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
#     )

#     cypher = """
#     MATCH (p:Person)
#     WHERE toLower(p.id) CONTAINS toLower($person_name)

#     OPTIONAL MATCH (p)-[r]->(n)
#     RETURN DISTINCT p, collect(DISTINCT {
#         rel: type(r),
#         target: tail(labels(n))[0],
#         value: CASE
#             WHEN type(r) IN ['HAS_URL', 'HAS_EMAIL'] THEN toLower(n.id)
#             ELSE n.id
#         END
#     }) AS related_info
#     """

#     with driver.session() as session:
#         results = session.run(cypher, person_name=person_name)
#         people = [result.data() for result in results]
#         if len(people) == 0:
#             return f"No people found with name containing '{person_name}'."
#         elif len(people) == 1:
#             person = people[0]
#             base = f"Found {person['p']['id']} ({person['p']['url']}) with related info:\n"
#             grouped = defaultdict(list)
#             for rel in person["related_info"]:
#                 key = (rel["rel"], rel["target"])
#                 grouped[key].append(rel["value"])

#             for (rel, target), values in grouped.items():
#                 joined = ", ".join(values)
#                 base += f"{target}(s): {joined}.\n"
#             return base
#         else:
#             base = f"Found {len(people)} people with name containing '{person_name}':\n"
#             for person in people:
#                 grouped = defaultdict(list)
#                 for rel in person["related_info"]:
#                     key = (rel["rel"], rel["target"])
#                     grouped[key].append(rel["value"])

#                 for (rel, target), values in grouped.items():
#                     joined = ", ".join(values)
#                     base += f"{target}(s): {joined}.\n"
#             return base


# @tool(args_schema=PersonFullProfileQuery)
# def get_person_full_profile(person_name) -> str:
#     """Retrieve a full profile of a person from the graph database."""
#     driver = GraphDatabase.driver(
#         os.getenv("NEO4J_URI"),
#         auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
#     )

#     cypher = """
#     MATCH (p:Person)-[:HAS_FULL_PROFILE]->(fp:Full_Profile)
#     WHERE toLower(p.name) CONTAINS toLower($person_name)
#     RETURN fp.content AS profile
#     """

#     with driver.session() as session:
#         result = session.run(cypher, person_name=person_name)
#         record = result.single()

#         if record and record.get("profile"):
#             return record["profile"]
#         else:
#             return f"No full profile found for '{person_name}'."


def get_existing_departments() -> set[str]:
    """Fetch the list of valid departments from the Neo4j database."""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    with driver.session() as session:
        result = session.run("MATCH (d:Department) RETURN d.id AS id")
        return {record["id"].lower() for record in result}


def validate_departments(
    input_departments: list[str],
) -> tuple[list[str], list[str]]:
    """Check if input departments exist in the graph.
    Returns (valid, invalid) lists."""
    existing = get_existing_departments()

    valid = [d for d in input_departments if d.lower() in existing]
    invalid = [d for d in input_departments if d.lower() not in existing]

    return valid, invalid


@tool(args_schema=DepartmentResearchInterestQuery)
def get_researchers_by_departments_and_interests(
    departments: list[str], interests: list[str], top_k: int = 10
) -> list[str]:
    """
    Query researchers by department and research interest.

    ✅ Use this tool only if the query explicitly mentions a department, school, or faculty.
    ❌ Do not use this tool if the user does not specify a department.

    Departments are matched to existing Department nodes in the database.
    If the department does not exist, this tool will return an error.

    Examples:
    - "Find AI researchers in the School of Engineering."
    - "Who in the Computer Science department works on NLP?"
    """

    # ✅ Validate input departments
    valid_depts, invalid_depts = validate_departments(departments)

    if not valid_depts:
        return [
            f"No valid departments found in the input: {', '.join(invalid_depts)}"
        ]

    if invalid_depts:
        return [
            f"Some departments are invalid and will be ignored: {', '.join(invalid_depts)}"
        ]

    # ✅ Embed research interests
    interest_embeddings = [embedding_model.embed_query(i) for i in interests]

    cypher = """
    UNWIND $interest_embeddings AS interest_emb
    CALL db.index.vector.queryNodes('research_interest_index', $topK, interest_emb)
    YIELD node AS ri, score AS interest_score

    MATCH (dept:Department)
    WHERE toLower(dept.id) IN $valid_departments

    MATCH (p:Person)-[:HAS_DEPARTMENT]->(dept)
    MATCH (p)-[:HAS_RESEARCH_INTEREST]->(ri)

    RETURN 
      p.name AS name, 
      p.url AS url, 
      dept.id AS department, 
      collect(DISTINCT ri.id) AS matched_interests,
      max(interest_score) AS best_score
    ORDER BY best_score DESC
    LIMIT $topK
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        results = session.run(
            cypher,
            valid_departments=[d.lower() for d in valid_depts],
            interest_embeddings=interest_embeddings,
            topK=top_k,
        )

        output = []
        for record in results:
            name = record.get("name")
            url = record.get("url")
            department = record.get("department")
            matched_interests = record.get("matched_interests", [])
            best_score = round(record.get("best_score", 0), 3)

            interests_str = (
                ", ".join(matched_interests)
                if matched_interests
                else "No matched interests"
            )

            output.append(
                f"{name} ({url}) - Department: {department} - Research Interests: {interests_str} (Best Score: {best_score})"
            )

    if not output:
        return [
            f"No researchers found matching departments {', '.join(valid_depts)} and interests {', '.join(interests)}."
        ]
    return output


@tool
def get_person_full_profile(url: str) -> str:
    """
    Retrieve the full live profile for a person by scraping their public webpage.

    ✅ Only use this tool if the user specifically provides a person's profile URL.
    ✅ The URL must belong to a Person node in the database. If not, the tool will refuse.

    This tool performs:
    - A check to ensure the URL belongs to a known person.
    - A live scrape of the content inside <div id="block-uos-public-content"> on the page.

    If the URL does not match anyone in the database, it will return an error.

    Example usage:
    - "Show me more information about Tony Prescott: https://www.sheffield.ac.uk/cs/people/academic/tony-prescott"
    """

    # 🔍 Validate the URL exists in the graph
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person {url: $url})
            RETURN p.name AS name
            """,
            url=url,
        )
        record = result.single()

    if not record:
        return f"❌ The URL '{url}' does not match any person in the database. Check aborted."

    person_name = record["name"]

    # 🌐 Fetch the page
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return (
                f"⚠️ The page for {person_name} ({url}) responded with HTTP {response.status_code}. "
                "Check if the page is broken or restricted."
            )

        # 🧠 Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find("div", id="block-uos-public-content")

        if content_div is None:
            return (
                f"✅ The page for {person_name} ({url}) is live (HTTP 200), "
                "but no content found in <div id='block-uos-public-content'>."
            )

        # Extract and clean the text
        content_text = content_div.get_text(separator="\n", strip=True)

        if not content_text.strip():
            return (
                f"✅ The page for {person_name} ({url}) is live (HTTP 200), "
                "but the content block appears empty."
            )

        return (
            f"✅ Live profile information for {person_name} ({url}):\n\n"
            f"{content_text}"
        )

    except requests.exceptions.RequestException as e:
        return (
            f"❌ Failed to connect to {url}. Error: {str(e)}. "
            f"This may indicate the page is down, blocked, or unreachable."
        )


@tool
def search_person_by_name(name_query: str, top_k: int = 3) -> list[str]:
    """
    Search for a person by name using vector similarity matching.

    ✅ Use this tool when a name search doesn't yield exact results.
    It finds the closest name matches from the graph database.

    Returns a list of names and URLs for top matching people.

    Example:
    - "Find someone named Grant Hill"
    """

    query_embedding = embedding_model.embed_query(name_query)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    cypher = """
    CALL db.index.vector.queryNodes('person_name_index', $top_k, $query_embedding)
    YIELD node AS person, score
    RETURN person.name AS name, person.url AS url, score
    ORDER BY score DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        results = session.run(
            cypher,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        output = [
            f"{record['name']} ({record['url']}) - Score: {round(record['score'], 3)}"
            for record in results
        ]

    if not output:
        return [f"No person found matching '{name_query}' in the database."]

    return output


@tool
def match_researchers_by_interests(
    interests: list[str], top_k: int = 10
) -> list[str]:
    """
    Find researchers who are linked to the given research interests (ignores departments).

    ✅ Use this tool when department constraints are unnecessary or unknown.

    Example:
    - "Find researchers related to 'artificial intelligence' and 'robotics'"
    """
    interest_embeddings = [embedding_model.embed_query(i) for i in interests]

    cypher = """
    UNWIND $interest_embeddings AS interest_emb
    CALL db.index.vector.queryNodes('research_interest_index', $topK, interest_emb)
    YIELD node AS ri, score AS interest_score

    MATCH (p:Person)-[:HAS_RESEARCH_INTEREST]->(ri)

    RETURN 
      p.name AS name, 
      p.url AS url, 
      collect(DISTINCT ri.id) AS matched_interests,
      max(interest_score) AS best_score
    ORDER BY best_score DESC
    LIMIT $topK
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        results = session.run(
            cypher,
            interest_embeddings=interest_embeddings,
            topK=top_k,
        )

        output = []
        for record in results:
            name = record.get("name")
            url = record.get("url")
            matched_interests = record.get("matched_interests", [])
            best_score = round(record.get("best_score", 0), 3)

            interests_str = (
                ", ".join(matched_interests)
                if matched_interests
                else "No matched interests"
            )

            output.append(
                f"{name} ({url}) - Research Interests: {interests_str} (Best Score: {best_score})"
            )

    if not output:
        return [
            f"⚠️ No researchers found matching interests: {', '.join(interests)}."
        ]
    return output


@tool
def list_departments() -> list[str]:
    """List all valid departments in the database."""
    return list(get_existing_departments())
