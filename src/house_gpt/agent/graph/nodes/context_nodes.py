import time
from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.states.house import AIHouseState
from house_gpt.core.logger import AppLogger

logger = AppLogger("Context-Node")

def context_injection_node(state: AIHouseState):
    t = time.time()
    schedule_context = ScheduleContextGenerator.get_current_activity()
    apply_activity = schedule_context != state.get("current_activity", "")
    logger.debug(f"[context_injection] apply={apply_activity} duration={time.time()-t:.3f}s")
    return {"apply_activity": apply_activity, "current_activity": schedule_context}