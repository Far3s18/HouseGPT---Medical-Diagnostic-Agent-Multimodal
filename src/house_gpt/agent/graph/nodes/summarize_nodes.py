from house_gpt.states.house import AIHouseState
from house_gpt.core.settings import settings
from house_gpt.agent.helpers.model_factory import get_small_model
from langchain_core.messages import HumanMessage, RemoveMessage
from house_gpt.core.logger import AppLogger

logger = AppLogger("Summarize-Node")

async def summarize_conversation_node(state: AIHouseState):
    from house_gpt.agent.helpers.model_factory import get_small_model
 
    t = time.time()
    model = get_small_model()
    summary = state.get("summary", "")
 
    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Dr House and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Dr House and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Dr House and the user:"
        )
 
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]
    ]
    logger.info(f"[summarize] messages_pruned={len(delete_messages)} duration={time.time()-t:.3f}s")
    return {"summary": response.content, "messages": delete_messages}