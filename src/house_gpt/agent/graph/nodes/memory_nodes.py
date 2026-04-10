import time
import asyncio
from house_gpt.core.logger import AppLogger
from house_gpt.memory.ltm.memory_manager import get_memory_manager
from house_gpt.states.house import AIHouseState
from house_gpt.agent.helpers.formatter import get_format_memories

_background_tasks: set[asyncio.Task] = set()
logger = AppLogger("Memory-Node")

async def memory_extraction_node(state: AIHouseState):
    t = time.time()
    if not state["messages"]:
        return {}

    task = asyncio.create_task(
        _safe_extract_memories(state["messages"][-1], state["user_id"])
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    logger.debug(f"[TIMER] Memory extraction: {time.time() - t:.2f}s")
    return {}

async def memory_injection_node(state: AIHouseState):
    t = time.time()
    memory_manager = get_memory_manager(state["user_id"])
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    loop = asyncio.get_running_loop()
    memories = await loop.run_in_executor(
        None, memory_manager.get_relevant_memories, recent_context
    )
    try:
        memories = await asyncio.wait_for(
            loop.run_in_executor(None, memory_manager.get_relevant_memories, recent_context),
            timeout=10
        )
    except asyncio.TimeoutError:
        logger.warning(f"Memory injection timed out for user={state['user_id']}, proceeding without memories")
        memories = []
    memory_context = get_format_memories(memories)
    logger.debug(f"[TIMER] memory injection: {time.time() - t:.2f}s | memories_found={len(memories)}")
    return {"memory_context": memory_context}


async def _safe_extract_memories(message, user_id: str):
    try:
        memory_manager = get_memory_manager(user_id)
        await memory_manager.extract_and_store_memories(message)
    except Exception as e:
        logger.error(f"Background memory extraction failed: {e}", exc_info=True)