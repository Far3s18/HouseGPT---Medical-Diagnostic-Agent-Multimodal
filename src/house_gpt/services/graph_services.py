import asyncio
from langchain_core.messages import HumanMessage
from house_gpt.core.graph_instance import get_graph
from house_gpt.core.logger import AppLogger

logger = AppLogger("GraphService")

GRAPH_TIMEOUT = 60

async def invoke_graph(message: str, session_id: str):
    compiled_graph = get_graph()
    try:
        await asyncio.wait_for(
            compiled_graph.ainvoke(
                {"messages": [HumanMessage(content=message)], "user_id": session_id},
                {"configurable": {"thread_id": session_id}},
            ),
            timeout=GRAPH_TIMEOUT
        )
        output_state = await compiled_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        return output_state
    except asyncio.TimeoutError:
        logger.error(f"Graph invocation timed out after {GRAPH_TIMEOUT}s for session={session_id}")
        raise