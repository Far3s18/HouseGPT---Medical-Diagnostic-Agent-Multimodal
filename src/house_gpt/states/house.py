from langgraph.graph import MessagesState

class AIHouseState(MessagesState):
    user_id: str
    summary: str | None
    workflow: str | None
    audio_buffer: bytes | None
    image_path: str | None
    current_activity: str
    apply_activity: bool
    memory_context: str
