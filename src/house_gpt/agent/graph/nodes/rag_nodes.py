from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from house_gpt.states.house import AIHouseState
from house_gpt.agent.chains import get_character_response_chain
from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.agent.helpers.formatter import build_rag_context
from house_gpt.memory.rag.rag_memory import get_medical_rag
from house_gpt.core.logger import AppLogger

logger = AppLogger("RAG")

async def medical_rag_node(state: AIHouseState, config: RunnableConfig):
    logger.info("Use RAG")
    query = state["messages"][-1].content
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")
    results = get_medical_rag().search_data(query=query, k=5)
    rag_context = build_rag_context(results)
    chain = get_character_response_chain(state.get("summary", ""))
    response = await chain.ainvoke({
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
            "medical_context": rag_context,
        },
        config=config
    )
    return {"messages": AIMessage(content=response), "medical_context": rag_context}