# 🧠 Sheffield Researcher RAG UI

A custom fork of OpenWebUI modified to support Graph Retrieval Augmented Generation (Graph RAG) across scraped researcher profiles from the University of Sheffield.

The project aims to enable natural language querying over academic profiles, providing contextual, AI driven answers to research related queries.

## ✨ Features

- **Graph RAG**: Query academic knowledge via an LLM-powered LangGraph agent  
- **Neo4j Knowledge Graph**: Stores relationships between people, departments, and research interests  
- **LangGraph Integration**: Intelligent routing of queries with context aware tool use  
- **Chat Persistence**: Async PostgreSQL based chat history storage  

## 🧰 Tech Stack

- **Frontend**: Custom fork of OpenWebUI  
- **Backend**: FastAPI with LangGraph agent logic  
- **LLM Interface**: Supports OpenAI, Ollama, and other backends  
- **GraphDB**: Neo4j for graph-based retrieval and reasoning  
- **Async Storage**: PostgreSQL with PostgresSaver for chat memory  

## 🚀 Getting Started

1. **Clone the repository**
```
git clone git@github.com:RSE-Sheffield/uos-grants.git
cd uos-grants
```

2. **Set Up Environment**
```
cp .env.example .env
```

3. **Start with Docker Compose**
```
docker compose up --build
```
Other versions of docker compose may require you to run
```
docker-compose up --build
```
You may also add the `-d` flag to run the stack in the background.

## ⚙️ Environment Variables

Environment variables needed to configure response generation, embedding, and database connection.

```
# Response model variables
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
EMBEDDING_NODES=Research_Interest, Department, Person
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

# 🧪 Usage

## 🗃️ Database and Graph Population
How profiles are fetched, stored in PostgreSQL, and used to build the Neo4j graph.

The population of both the PostgreSQL and Neo4j databases is fully automated. This ensures that researcher information is regularly collected, structured, and updated without manual intervention.

### 🏗️ Initial Setup
Steps to scrape data, extract structured information, and build the graph.

1. Sitemap Scraping
The system begins by fetching the University of Sheffield sitemap. All URLs containing /people/ are extracted as candidate staff profile pages.

2. Profile Extraction and Storage
Each profile page is scraped, and the following fields are collected where available:

  - Full name

  - Contact details (email, phone, address)
 
  - School or department

  - Research interests

  - Full profile text

  - Last modified date (from the sitemap XML)

  - These are stored in a PostgreSQL database, with the last_modified timestamp used to track changes over time.

3. Graph Construction in Neo4j
After all profiles are scraped, the system builds a Neo4j graph:

  - A Person node is created for each staff member, with attributes like name and URL.

  - Related entities (e.g., School, Role, Email, Address, Telephone) are created as individual nodes and connected to the person node.

  - Research interests (if present) are passed to a configurable LLM to extract individual topics.

  - Each unique research interest is stored as a Research_Interest node and linked to the corresponding staff member(s).

  - The graph ensures node reuse, so duplicate schools or shared interests are only created once and reused via relationships.

4. Embedding Generation
Nodes for Person, School, and Research_Interest are embedded using a configurable embedding model. These embeddings are used for semantic search and retrieval when querying the graph.

### 🔄 Ongoing Updates

How the system stays up-to-date with periodic re-scraping and graph updates.
Z
To keep the data current:

- The sitemap is periodically re-fetched.

- For each /people/ link, the last_modified value is compared to the stored value in PostgreSQL.

- If the timestamps differ:

  - The profile is re-scraped.

  - The corresponding Person node and its direct relationships are deleted and rebuilt in Neo4j using the same logic as the initial setup.

- Updates are automatically triggered:

  - On Docker Compose startup

  - Periodically while the stack is running

This ensures the graph remains accurate and up-to-date with minimal human intervention.

### 🧑‍💻 Using the UI

How to interact with the system via natural language queries.

- Navigate to `http://localhost`
- Enter a research-related query such as:
    > "Which researchers work in sustainable energy?"
- Responses are generated based on:
  - Matching research interests
  - Graph traversal using relationships in the Neo4j knowledge graph.
  - Reasoning and response by the LangGraph agent powered by the configured LLM.

## 🛠️ Model Configuration

Explanation of configurable LLMs and embedding providers using LangChain.

The system supports fully configurable LLM and embedding model providers via [LangChain integrations](https://docs.langchain.com). This allows you to easily switch between providers and models depending on your use case, budget, or availability.

### 🧾 Supported Providers

Table of all compatible providers and their LangChain integration keys.

The following providers are currently supported:

| Provider Key               | LangChain Integration               |
|----------------------------|-------------------------------------|
| `openai`                   | `langchain-openai`                  |
| `anthropic`                | `langchain-anthropic`               |
| `azure_openai`             | `langchain-openai`                  |
| `azure_ai`                 | `langchain-azure-ai`                |
| `google_vertexai`          | `langchain-google-vertexai`         |
| `google_genai`             | `langchain-google-genai`            |
| `bedrock`                  | `langchain-aws`                     |
| `bedrock_converse`         | `langchain-aws`                     |
| `cohere`                   | `langchain-cohere`                  |
| `fireworks`                | `langchain-fireworks`               |
| `together`                 | `langchain-together`                |
| `mistralai`                | `langchain-mistralai`               |
| `huggingface`              | `langchain-huggingface`            |
| `groq`                     | `langchain-groq`                    |
| `ollama`                   | `langchain-ollama`                  |
| `google_anthropic_vertex`  | `langchain-google-vertexai`         |
| `deepseek`                 | `langchain-deepseek`                |
| `ibm`                      | `langchain-ibm`                     |
| `nvidia`                   | `langchain-nvidia-ai-endpoints`     |
| `xai`                      | `langchain-xai`                     |
| `perplexity`               | `langchain-perplexity`             |

### 🎯 Model Selection

How to choose specific models for your use case.

Each provider supports one or more models. You can configure models by setting the appropriate model name string. For example:

- To use OpenAI’s GPT-4o Mini: `gpt-4o-mini`
- To use Google’s Gemini 2.5 Pro: `gemini-2.5-pro`

Refer to the specific provider’s documentation for a full list of supported model variants.

### 🔐 Authentication

How to securely provide API keys for model access.

Your provider-specific API key should be supplied via the relevant `API_KEY` environment variable. This key is used to authenticate all model requests.

### 🧩 Model Roles

Defines the distinct responsibilities of each model:

Three distinct models can be configured:

1. **LLM Response Model**  
   Used to generate responses to user queries.

2. **Embedding Model**  
   Generates embedding vectors for nodes and incoming queries to support semantic search and retrieval.

3. **Graph Generation Model**  
   Processes staff profile texts to extract research interests via an LLM, which are then structured into the Neo4j graph.

Each model can be independently configured to use different providers and model variants.
