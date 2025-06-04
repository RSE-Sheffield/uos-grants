from .tools import (
    research_interests_query,
    get_people_by_name,
    get_person_full_profile,
    get_researchers_by_departments_and_interests
)
from langgraph.prebuilt import create_react_agent

prompt = """You are an AI assistant that acts as a grant matcher and directory for the University of Sheffield. "
            Based on the context provided, list all the people found. If the query is about a specific person respond to that to the best of your ability. "
            For each person, include their name, department, website, and contact info. Expand where possible. 
            If a query asks for your opinion, form one using only the context. Do not make up information.
            If the query is about research interests, provide a list of research interests for the person or department.
            If the query is about a specific person, provide their full profile including all related information.
            If the query is about researchers in specific departments or with specific research interests, provide a list of researchers that match the criteria.
            Do not use any external knowledge, only use the context provided."""

tools = [
    research_interests_query,
    get_people_by_name,
    get_person_full_profile,
    get_researchers_by_departments_and_interests
]

def get_react_agent(model, tools=tools, prompt=prompt, memory=None):
    """
    Create a REACT agent with the given model, tools, and prompt.

    Args:
        model: The language model to use.
        tools: A list of tools that the agent can use.
        prompt: The prompt to use for the agent.
        memory: Optional memory for the agent.

    Returns:
        A REACT agent instance.
    """
    
    if memory is None:
        return create_react_agent(model, tools, prompt=prompt)
    else:
        return create_react_agent(
            model, tools, prompt=prompt, checkpointer=memory
        )
