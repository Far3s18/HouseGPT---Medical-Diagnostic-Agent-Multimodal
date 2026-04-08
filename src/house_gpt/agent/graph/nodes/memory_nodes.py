from house_gpt.memory.ltm.memory_manager import get_memory_manager
from house_gpt.states.house import AIHouseState
from house_gpt.agent.helpers.formatter import get_format_memories


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

    memory_context = get_format_memories(memories)

    return {"memory_context": memory_context}
