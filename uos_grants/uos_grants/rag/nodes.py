from typing import Annotated, TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from uos_grants.utils import trim_message_history


# === LangGraph State Definition ===
class State(TypedDict):
    messages: Annotated[List, add_messages]
    should_retrieve: bool
    intent: str


# === Basic Pass-Through Nodes ===

def reuse_previous_context(state: State) -> State:
    return {
        "messages": state["messages"],
        "should_retrieve": False,
        "intent": state.get("intent", "search_by_topic")
    }


# === Intent Classification Node ===

def classify_intent_factory(llm):
    def classify_intent(state: State) -> State:
        latest_query = next(
            msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "Classify the user query below as one of the following:\n\n"
                "- search_by_topic: searching based on a topic or research interest\n"
                "- lookup_by_name: asking about a specific researcher by name\n"
                "- compare: comparing departments, researchers, or areas\n"
                "- chitchat: small talk, questions about you, or other general questions\n\n"
                "Respond ONLY with the label."
            )),
            latest_query
        ])
        intent = llm.invoke(prompt.format_messages()).content.strip().lower()
        return {
            "messages": state["messages"],
            "should_retrieve": False,
            "intent": intent
        }
    return classify_intent


# === Retrieval Decision Node ===

def decide_retrieval_factory(llm):
    def decide_retrieval(state: State) -> State:
        latest_query = next(
            msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
        )
        previous_context = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, AIMessage) and "Source:" in msg.content
        ]
        previous_context = "\n\n".join(previous_context)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                """If there is sufficient context in this to supply the last query with the information it needs reply 'do_not_retrieve'.
                If you think there is not enough context, reply 'retrieve'. Reply ONLY with 'retrieve' or 'do_not_retrieve'."""
            ),
            HumanMessage(
                content=f"Most recent query: {latest_query.content}\nPrevious context count: {previous_context}"
            ),
        ])
        decision = llm.invoke(prompt.format_messages()).content.strip().lower()
        return {
            "messages": state["messages"],
            "should_retrieve": decision == "retrieve",
            "intent": state.get("intent", "search_by_topic")
        }

    return decide_retrieval


# === Retriever Wrapper ===

def make_retriever_factory(retriever):
    def custom_retrieve(query: str) -> str:
        k = 10
        docs = retriever.similarity_search(query, k=k)
        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
        )
    return custom_retrieve


def retrieve_context_factory(custom_retrieve_fn):
    def retrieve_context(state: State) -> State:
        query = next(
            msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
        ).content
        results = custom_retrieve_fn(query)
        return {
            "messages": state["messages"] + [AIMessage(content=results)],
            "should_retrieve": False,
            "intent": state["intent"],
        }

    return retrieve_context


# === Lookup by Name ===

def lookup_by_name_factory(retriever):
    def lookup_by_name(state: State) -> State:
        query = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)).content
        results = retriever.similarity_search(query, k=10, filter={"name": query})
        if not results:
            results = retriever.similarity_search(query, k=5)  # fuzzy fallback

        text = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in results
        )
        return {
            "messages": state["messages"] + [AIMessage(content=text)],
            "should_retrieve": False,
            "intent": state["intent"]
        }

    return lookup_by_name


# === Compare Node ===

def compare_factory(retriever, llm):
    def compare(state: State) -> State:
        query = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)).content
        results = retriever.similarity_search(query, k=10)

        merged_context = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in results
        )

        system_content = (
            "You are an assistant comparing multiple researchers based on the query. "
            "Compare their research areas, departments, and relevance to the user’s request. "
            "Only use the provided context:\n\n" + merged_context
        )
        prompt = [
            SystemMessage(content=system_content),
            HumanMessage(content=query)
        ]
        response = llm.invoke(prompt)

        return {
            "messages": state["messages"] + [response],
            "should_retrieve": False,
            "intent": state["intent"]
        }
    return compare


# === Chitchat / Fallback Node ===

def chitchat_factory(llm):
    def chitchat(state: State) -> State:
        query = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
        prompt = [SystemMessage("You are a helpful assistant."), query]
        response = llm.invoke(prompt)
        return {
            "messages": state["messages"] + [response],
            "should_retrieve": False,
            "intent": state["intent"]
        }
    return chitchat


# === Response Generation ===

def generate_response_factory(llm):
    async def generate_response(state: State):
        from langchain_core.messages import AIMessage, AIMessageChunk

        state['messages'] = trim_message_history(state['messages'])
        human_msg = next(
            msg for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        )

        context_messages = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and "Source:" in msg.content
        ]
        context = "\n\n".join(context_messages)

        system_content = (
            "You are an AI assistant that acts as a directory for the University of Sheffield. "
            "Based on the context provided, list all the people found. If the query is about a specific person respond to that to the best of your ability. "
            "For each person, include their name, department, website, and contact info. Expand where possible. "
            "If a query asks for your opinion, form one using only the context. Do not make up information.\n\nContext:\n" + context
        )

        prompt = [SystemMessage(content=system_content), human_msg]

        chunks = []
        async for chunk in llm.astream(prompt):
            if isinstance(chunk, AIMessageChunk):
                chunks.append(chunk.content)
                yield {"stream": chunk.content}

        final = AIMessage(content="".join(chunks))
        yield {
            "messages": state["messages"] + [final],
            "should_retrieve": False,
            "intent": state["intent"]
        }

    return generate_response


# === Graph Assembly ===

def make_graph(llm, retriever, memory=None):
    custom_retrieve = make_retriever_factory(retriever)
    graph = StateGraph(State)

    # Add Nodes
    graph.add_node("classify_intent", classify_intent_factory(llm))
    graph.add_node("decide_retrieval", decide_retrieval_factory(llm))
    graph.add_node("retrieve_context", retrieve_context_factory(custom_retrieve))
    graph.add_node("reuse_previous_context", reuse_previous_context)
    graph.add_node("generate_response", generate_response_factory(llm))
    graph.add_node("lookup_by_name", lookup_by_name_factory(retriever))
    graph.add_node("compare", compare_factory(retriever, llm))
    graph.add_node("chitchat", chitchat_factory(llm))

    # Entry point
    graph.set_entry_point("classify_intent")

    # Intent routing
    graph.add_conditional_edges(
        "classify_intent",
        lambda s: s["intent"],
        {
            "search_by_topic": "decide_retrieval",
            "lookup_by_name": "lookup_by_name",
            "compare": "compare",
            "chitchat": "chitchat",
        }
    )

    # Retrieval logic
    graph.add_conditional_edges(
        "decide_retrieval",
        lambda s: "retrieve_context" if s["should_retrieve"] else "reuse_previous_context",
        {
            "retrieve_context": "retrieve_context",
            "reuse_previous_context": "reuse_previous_context",
        }
    )

    # All paths lead to generate_response
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("reuse_previous_context", "generate_response")
    graph.add_edge("lookup_by_name", "generate_response")
    graph.add_edge("compare", "generate_response")
    graph.add_edge("chitchat", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=memory)
