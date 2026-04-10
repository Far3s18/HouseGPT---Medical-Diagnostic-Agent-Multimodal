from house_gpt.states.house import AIHouseState
from house_gpt.core.settings import settings
from typing import Literal
from langgraph.graph import END

def should_summarize_conversation(state: AIHouseState,) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]
    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"
    return END

def select_workflow(state: AIHouseState) -> Literal["conversation_node", "medical_rag_node"]:
    workflow = state["workflow"]
    if workflow == "conversation":
        return "conversation_node"
    elif workflow == "rag":
        return "medical_rag_node"
    else:
        return END