from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.states.house import AIHouseState
from house_gpt.core.logger import AppLogger
import time

logger = AppLogger("Context-Injection")

def context_injection_node(state: AIHouseState):
    t = time.time()
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    logger.debug(f"[TIMER] Context injection: {time.time() - t:.2f}s")
    return {"apply_activity": apply_activity, "current_activity": schedule_context}