from langgraph.graph import StateGraph, START, END
from functools import lru_cache
from house_gpt.states.house import AIHouseState
from house_gpt.agent.graph.edges import select_workflow, should_summarize_conversation
from house_gpt.agent.graph.nodes import (
    memory_extraction_node,
    router_node,
    context_injection_node,
    memory_injection_node,
    conversation_node,
    summarize_conversation_node
)

@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AIHouseState)

    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    graph_builder.add_edge(START, "memory_extraction_node")
    graph_builder.add_edge("memory_extraction_node", "router_node")
    graph_builder.add_edge("router_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")
    graph_builder.add_conditional_edges("memory_injection_node", select_workflow)
    graph_builder.add_conditional_edges("conversation_node", should_summarize_conversation)
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder

def get_graph_builder():
    graph = create_workflow_graph().compile()
    return graph