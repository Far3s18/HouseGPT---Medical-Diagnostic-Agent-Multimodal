import asyncio
import time
from house_gpt.core.logger import AppLogger
from house_gpt.memory.ltm.memory_manager import get_memory_manager
from house_gpt.states.house import AIHouseState
from house_gpt.agent.helpers.formatter import get_format_memories

_background_tasks: set[asyncio.Task] = set()
logger = AppLogger("Memory_nodes")

async def memory_extraction_node(state: AIHouseState):
    t = time.time()
    if not state["messages"]:
        return {}

    task = asyncio.create_task(
        _safe_extract_memories(state["messages"][-1], state["user_id"])
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    print(f"[TIMER] memory_extraction_node: {time.time() - t:.2f}s")
    return {}

def memory_injection_node(state: AIHouseState):
    t = time.time()
    memory_manager = get_memory_manager(state["user_id"])

    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    memory_context = get_format_memories(memories)
    print(f"[TIMER] inject_node: {time.time() - t:.2f}s")
    return {"memory_context": memory_context}


async def _safe_extract_memories(message, user_id: str):
    try:
        memory_manager = get_memory_manager(user_id)
        await memory_manager.extract_and_store_memories(message)
    except Exception as e:
        logger.error(f"Background memory extraction failed: {e}", exc_info=True)