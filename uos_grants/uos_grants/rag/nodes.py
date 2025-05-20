# uos_grants/rag/nodes.py


from typing import Annotated, TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)

from uos_grants.utils import trim_message_history

# === LangGraph State Definition ===
class State(TypedDict):
    messages: Annotated[List, add_messages]
    should_retrieve: bool


# === Agent Node (pass through) ===
def agent(state: State) -> State:
    """
    Agent node that passes the state through without modification.
    This node is used to keep the messages as-is without any changes.

    Args:
        state (State): The current state of the conversation.

    Returns:
        State: The state with messages unchanged.
    """
    return {"messages": state["messages"]}


def reuse_previous_context(state: State) -> State:
    return {
        "messages": state["messages"],
        "should_retrieve": False,
    }


def inject_human_message(state: State) -> State:
    # Assume the last message passed in was the user input
    latest_human = next(
        msg
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )
    return {
        "messages": state["messages"] + [latest_human],
        "should_retrieve": False,
    }


def make_retriever_factory(retriever):
    def custom_retrieve(query: str) -> str:
        k = 10
        docs = retriever.similarity_search(query, k=k)
        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )

    return custom_retrieve


def generate_response_factory(llm):
    def generate_response(state: State) -> State:
        # Extract most recent HumanMessage
        state['messages'] = trim_message_history(state['messages'])
        human_msg = next(
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        )

        # Only get the latest block of context-containing AI messages (you could optimize this)
        context_messages = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, AIMessage) and "Source:" in msg.content
        ]
        context = "\n\n".join(context_messages)

        # Build final prompt with context injected directly
        #system_content = (
        #    "You are an AI assistant that acts as a directory for the University of Sheffield."
        #    "Based solely on the context provided, list all the people found. "
        #    "For each person, include their name, department, website, and contact info. "
        #    "Do not make up information.\n\nContext:\n" + context
        #)
        system_content = (
            "You are an AI assistant that acts as a directory for the University of Sheffield."
            "Based on the context provided, list all the people found. If the query is about a specific person respond to that to the best of your ability."
            "For each person, include their name, department, website, and contact info. If you're asked to exapand on individuals do that using the context provided about them."
            "If a query asks for your opinion, use the context provided to form one."
            "Do not make up information.\n\nContext:\n" + context
        )
        prompt = [SystemMessage(content=system_content), human_msg]
        response = llm.invoke(prompt)

        return {
            "messages": state["messages"] + [response],
#            "should_retrieve": False,
        }

    return generate_response


def decide_retrieval_factory(llm):
    def decide_retrieval(state: State) -> State:
        latest_query = next(
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        )
        print("Latest query:", latest_query.content)
        previous_context = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, AIMessage) and "Source:" in msg.content
        ]
        previous_context = "\n\n".join(previous_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    """If there is sufficient context in this to supply the last query with the information it needs reply 'do_not_retrieve'.
                        If you think there is not enough context, reply 'retrieve'. If there is no relevent context for the most recent query, reply 'retrieve'.
                        Reply ONLY with 'retrieve' or 'do_not_retrieve'.
                        If there is no information relevant for the most recent query, reply 'retrieve'."""
                ),
                HumanMessage(
                    content=f"Most recent query: {latest_query.content}\nPrevious context count: {previous_context}"
                ),
            ]
        )
        print(latest_query.content)
        decision = llm.invoke(prompt.format_messages()).content.strip().lower()
        print("Decision:", decision)

        return {
            "messages": state["messages"],
            "should_retrieve": decision == "retrieve",
        }

    return decide_retrieval


def retrieve_context_factory(custom_retrieve_fn):
    def retrieve_context(state: State) -> State:
        query = next(
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ).content
        results = custom_retrieve_fn(query)
        return {
            "messages": state["messages"] + [AIMessage(content=results)],
            "should_retrieve": False,
        }

    return retrieve_context


def make_graph(llm, retriever, memory=None):

    custom_retrieve = make_retriever_factory(retriever)

    graph = StateGraph(State)
    graph.add_node("decide_retrieval", decide_retrieval_factory(llm))
    graph.add_node(
        "retrieve_context", retrieve_context_factory(custom_retrieve)
    )
    graph.add_node("reuse_previous_context", reuse_previous_context)
    graph.add_node("generate_response", generate_response_factory(llm))

    graph.set_entry_point("decide_retrieval")
    graph.add_conditional_edges(
        "decide_retrieval",
        lambda s: (
            "retrieve_context"
            if s["should_retrieve"]
            else "reuse_previous_context"
        ),
        {
            "retrieve_context": "retrieve_context",
            "reuse_previous_context": "reuse_previous_context",
        },
    )
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("reuse_previous_context", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=memory)
