from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from house_gpt.agent.graph import create_workflow_graph
from house_gpt.core.settings import settings

async def invoke_graph(message: str, session_id: str):
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as memory:
        graph = create_workflow_graph().compile(checkpointer=memory)
        await graph.ainvoke(
            {"messages": [HumanMessage(content=message)], "user_id": session_id},
            {"configurable": {"thread_id": session_id}},
        )
        output_state = await graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
    return output_state
