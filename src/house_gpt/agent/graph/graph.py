from langgraph.graph import StateGraph, START, END
from functools import lru_cache
from pathlib import Path
from langchain_core.messages import HumanMessage
from house_gpt.states.house import AIHouseState
from house_gpt.agent.graph.edges import select_workflow, should_summarize_conversation
from house_gpt.agent.graph.nodes import (
    memory_extraction_node,
    router_node,
    context_injection_node,
    memory_injection_node,
    conversation_node,
    summarize_conversation_node,
    medical_rag_node,
    dispatch_node
)

@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AIHouseState)

    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("dispatch_node", dispatch_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("medical_rag_node", medical_rag_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    graph_builder.add_edge(START, "memory_extraction_node")
    graph_builder.add_edge(START, "router_node")
    graph_builder.add_edge(START, "context_injection_node")
    graph_builder.add_edge(START, "memory_injection_node")
    graph_builder.add_edge("memory_extraction_node", "dispatch_node")
    graph_builder.add_edge("router_node", "dispatch_node")
    graph_builder.add_edge("context_injection_node", "dispatch_node")
    graph_builder.add_edge("memory_injection_node", "dispatch_node")

    graph_builder.add_conditional_edges("dispatch_node", select_workflow)
    graph_builder.add_conditional_edges("conversation_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("medical_rag_node", should_summarize_conversation)

    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder

def _save_graph_image(graph, filename: str = "house_gpt_graph.png") -> None:
    try:
        output_dir = Path("images")
        output_dir.mkdir(exist_ok=True)
        image_bytes = graph.get_graph().draw_mermaid_png()
        output_path = output_dir / filename
        output_path.write_bytes(image_bytes)
        logger.info(f"[Graph] Diagram saved to {output_path}")
    except Exception as e:
        logger.warning(f"[Graph] Could not save graph image (non-fatal): {e}")


def get_graph_builder():
    graph = create_workflow_graph().compile()
    _save_graph_image(graph)
    return graph