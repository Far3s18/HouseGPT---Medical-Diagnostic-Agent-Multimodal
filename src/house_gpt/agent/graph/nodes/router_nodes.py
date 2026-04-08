import os
from house_gpt.states.house import AIHouseState
from house_gpt.agent.chains import get_router_chain
from house_gpt.core.settings import settings


async def router_node(state: AIHouseState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    return {"workflow": response.response_type}