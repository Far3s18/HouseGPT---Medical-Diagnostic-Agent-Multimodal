import os
from house_gpt.states.house import AIHouseState
from house_gpt.agent.helpers.model_factory import get_small_model, get_large_model
from house_gpt.agent.chains import get_router_chain, get_character_response_chain
from house_gpt.memory.ltm.memory_manager import get_memory_manager
from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.core.settings import settings
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig


async def router_node(state: AIHouseState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    return {"workflow": response.response_type}

async def memory_extraction_node(state: AIHouseState):
    if not state["messages"]:
        return {}

    memory_manager = get_memory_manager(state["user_id"])
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}

def memory_injection_node(state: AIHouseState):
    memory_manager = get_memory_manager(state["user_id"])

    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}

def context_injection_node(state: AIHouseState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    return {"apply_activity": apply_activity, "current_activity": schedule_context}

async def conversation_node(state: AIHouseState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke({
        "messages": state["messages"],
        "current_activity": current_activity,
        "memory_context": memory_context
    }, config=config)

    return {"messages": AIMessage(content=response)}

async def summarize_conversation_node(state: AIHouseState):
    model = get_small_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Dr House and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Dr House and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Dr House and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    
    return {"summary": response.content, "messages": delete_messages}