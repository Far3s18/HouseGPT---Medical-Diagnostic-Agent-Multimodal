from langchain_core.messages import HumanMessage
from house_gpt.core.graph_instance import get_graph

async def invoke_graph(message: str, session_id: str):
    compiled_graph = get_graph()
    await compiled_graph.ainvoke(
        {"messages": [HumanMessage(content=message)], "user_id": session_id},
        {"configurable": {"thread_id": session_id}},
    )
    output_state = await compiled_graph.aget_state(
        config={"configurable": {"thread_id": session_id}}
    )
    return output_state
