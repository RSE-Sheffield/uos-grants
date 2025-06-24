# Sheffield Researcher RAG UI

A custom fork of OpenWebUI modified to support Graph Retrieval Augmented Generation (Graph RAG) across scraped researcher profiles from the University of Sheffield.

The project aims to enable natural language querying over academic profiles, providing contextual, AI driven answers to research related queries.

## Features

Graph RAG: Query academic knowledge via an LLM-powered LangGraph agent
Neo4j Knowledge Graph: Stores relationships between people, departments, and research interests
LangGraph Integration: Intelligent routing of queries with context aware tool use
Chat Persistence: Async PostgreSQL based chat history storage

## Tech Stack

- Frontend: Custom fork of OpenWebUI
- Backend: FastAPI with LangGraph agent logic
- LLM Interface: Supports OpenAI, Ollama, and other backends
- GraphDB: Neo4j for graph-based retrieval and reasoning
- Async Storage: PostgreSQL with PostgresSaver for chat memory

## Getting Started

1. Clone the repository
```
git clone git@github.com:RSE-Sheffield/uos-grants.git
cd uos-grants
```

2. Set Up Environment
```
cp .env.example .env
```

3. Start with Docker Compose
```
docker compose up --build
```
Other versions of docker compose may require you to run
```
docker-compose up --build
```
You may also add the `-d` flag to run the stack in the background.

## Environment Variables
You will need to configure the following environment variables according to your setup.

```
# LLM Model Variables
LLM_MODEL_PROVIDER=openai
LLM_MODEL=gpt-4.1-nano-2025-04-14
LLM_API_KEY=sk-...

# Neo4j graph rag generation variables
GRAPH_LLM_PROVIDER=openai
GRAPH_LLM_MODEL=gpt-4.1-nano-2025-04-14
GRAPH_LLM_API_KEY=sk-...

# Embedding model variables
EMBEDDING_MODEL_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
EMBEDDING_MODEL_API_KEY=sk-...
```

The following environment variables are configured in the docker-compose.yaml file for the open-webui container, and should align with the setup of your postgres and neo4j setups.

```
# Database variables, should align with the postgres container variables.
DATABASE_URL: postgresql://user:pass@postgres:5432/uos_grants
CHAT_MEMORY_DB_URI: postgresql://user:pass@postgres:5432/uos_grants

# Neo4j variables, should align with the neo4j container variables.
NEO4J_URI: bolt://neo4j:7685
NEO4J_USERNAME: neo4j
NEO4J_PASSWORD: your_neo4j_password
```

## Usage

- Navigate to `http://localhost`
- Enter a research-related query such as:
> "Which researchers work in sustainable energy?"
- Responses are generated based on:
 - Matching research interests
 - Graph traversal using relationships in the Neo4j knowledge graph.
 - Reasoning and response by the LangGraph agent powered by the configured LLM.