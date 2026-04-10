import time
import asyncio
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from house_gpt.states.house import AIHouseState
from house_gpt.agent.chains import get_character_response_chain
from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.agent.helpers.formatter import build_rag_context
from house_gpt.memory.rag.rag_memory import get_medical_rag
from house_gpt.core.logger import AppLogger

logger = AppLogger("RAG-Node")

async def medical_rag_node(state: AIHouseState, config: RunnableConfig):
    t = time.time()
    query = state["messages"][-1].content
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")
 
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, get_medical_rag().search_data, query, 5)
    rag_context = build_rag_context(results)
 
    logger.info(f"[rag] results_found={len(results)} duration_so_far={time.time()-t:.3f}s")
 
    chain = get_character_response_chain(state.get("summary", ""))
    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
            "medical_context": rag_context,
        },
        config=config,
    )
    logger.info(f"[rag] total_duration={time.time()-t:.3f}s")
    return {"messages": AIMessage(content=response), "medical_context": rag_context}