import asyncio
import time
from langchain_core.messages import HumanMessage
from house_gpt.core.graph_instance import get_graph, get_pool
from house_gpt.core.logger import AppLogger

logger = AppLogger("GraphService")

GRAPH_TIMEOUT = 55
MAX_CONCURRENT = 8
MAX_QUEUE_DEPTH = 16
DAILY_LIMIT = 10

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_depth = 0 


async def check_and_increment_quota(user_id: str) -> bool:
    pool = get_pool()
    today = time.strftime("%Y-%m-%d")
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO user_daily_quota (user_id, day, request_count)
            VALUES (%s, %s, 1)
            ON CONFLICT (user_id, day)
            DO UPDATE SET request_count = user_daily_quota.request_count + 1
            RETURNING request_count
            """,
            (user_id, today),
        )
        row = await cursor.fetchone()
    return row[0] <= DAILY_LIMIT


async def _decrement_quota(user_id: str):
    try:
        pool = get_pool()
        today = time.strftime("%Y-%m-%d")
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE user_daily_quota
                   SET request_count = GREATEST(request_count - 1, 0)
                 WHERE user_id = %s AND day = %s
                """,
                (user_id, today),
            )
    except Exception:
        pass


async def invoke_graph(message: str, session_id: str) -> dict:
    global _queue_depth

    compiled_graph = get_graph()
    logger.info(f"[GRAPH] Invoking session={session_id} message_preview={message[:60]!r}")

    if _queue_depth >= MAX_QUEUE_DEPTH:
        logger.warning(f"[GRAPH][QUEUE_FULL] session={session_id} depth={_queue_depth}")
        raise RuntimeError("SERVER_BUSY")

    _queue_depth += 1
    try:
        acquired = await asyncio.wait_for(
            _semaphore.acquire(), timeout=GRAPH_TIMEOUT - 5
        )
    except asyncio.TimeoutError:
        logger.error(f"[GRAPH][WAIT_TIMEOUT] session={session_id} never got slot")
        raise RuntimeError("SERVER_BUSY")
    finally:
        _queue_depth -= 1

    try:
        output_state = await asyncio.wait_for(
            compiled_graph.ainvoke(
                {"messages": [HumanMessage(content=message)], "user_id": session_id},
                {"configurable": {"thread_id": session_id}},
            ),
            timeout=GRAPH_TIMEOUT,
        )
        logger.info(
            f"[GRAPH][OK] session={session_id} "
            f"workflow={output_state.get('workflow', 'conversation')}"
        )
        return output_state

    except asyncio.TimeoutError:
        logger.error(f"[GRAPH][TIMEOUT] session={session_id} exceeded {GRAPH_TIMEOUT}s")
        raise

    finally:
        _semaphore.release()