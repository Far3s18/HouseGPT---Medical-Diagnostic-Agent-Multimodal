from house_gpt.schedules.context_generation import ScheduleContextGenerator
from house_gpt.states.house import AIHouseState


def context_injection_node(state: AIHouseState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    return {"apply_activity": apply_activity, "current_activity": schedule_context}