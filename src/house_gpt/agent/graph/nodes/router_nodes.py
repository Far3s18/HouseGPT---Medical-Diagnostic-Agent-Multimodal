import os
import time
from house_gpt.states.house import AIHouseState
from house_gpt.agent.chains import get_router_chain
from house_gpt.core.settings import settings
from house_gpt.core.logger import AppLogger

logger = AppLogger("Router-Node")

async def router_node(state: AIHouseState):
    t = time.time()
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    print(f"[TIMER] router_node: {time.time() - t:.2f}s")
    return {"workflow": response.response_type}