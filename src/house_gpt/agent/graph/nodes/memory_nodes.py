import time
import asyncio
from house_gpt.core.logger import AppLogger
from house_gpt.memory.ltm.memory_manager import get_memory_manager
from house_gpt.states.house import AIHouseState
from house_gpt.agent.helpers.formatter import get_format_memories
import concurrent.futures


_qdrant_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="qdrant"
)

_background_tasks: set[asyncio.Task] = set()
logger = AppLogger("Memory-Nodes")

async def memory_extraction_node(state: AIHouseState):
    t = time.time()
    if not state["messages"]:
        return {}
    task = asyncio.create_task(
        _safe_extract_memories(state["messages"][-1], state["user_id"])
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    logger.debug(f"[memory_extraction] background task queued duration={time.time()-t:.3f}s")
    return {}

async def memory_injection_node(state: AIHouseState):
    t = time.time()
    user_id = state["user_id"]
    memory_manager = get_memory_manager(user_id)
    recent_context = " ".join(
        [m.content for m in state["messages"][-3:] if hasattr(m, "content")]
    )
    loop = asyncio.get_running_loop()
    try:
        memories = await asyncio.wait_for(
            loop.run_in_executor(
                _qdrant_executor,
                memory_manager.get_relevant_memories,
                recent_context
            ),
            timeout=10,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[memory_injection][TIMEOUT] user={user_id} proceeding without memories")
        memories = []
 
    memory_context = get_format_memories(memories)
    logger.info(
        f"[memory_injection] user={user_id} memories_found={len(memories)} duration={time.time()-t:.3f}s"
    )
    return {"memory_context": memory_context}


async def _safe_extract_memories(message, user_id: str):
    try:
        memory_manager = get_memory_manager(user_id)
        await memory_manager.extract_and_store_memories(message)
        logger.debug(f"[memory_extraction][BG] user={user_id} done")
    except Exception as e:
        logger.error(f"[memory_extraction][BG][ERROR] user={user_id} error={e}", exc_info=True)
 