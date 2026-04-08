
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from house_gpt.states.house import AIHouseState
from house_gpt.agent.chains import get_character_response_chain
from house_gpt.schedules.context_generation import ScheduleContextGenerator


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