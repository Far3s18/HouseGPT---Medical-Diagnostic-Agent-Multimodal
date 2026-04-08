from fastapi import APIRouter, Request, Response
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from house_gpt.agent.graph.graph import create_workflow_graph
from fastapi.responses import JSONResponse
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings

logger = AppLogger("API")
chat_router = APIRouter()

@chat_router.post("/chat")
async def chat_handler(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "")

        if not message:
            return JSONResponse(content={"error": "Message is required"}, status_code=400)
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)

        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as memory:
            graph = create_workflow_graph().compile(checkpointer=memory)
            await graph.ainvoke(
                {"messages": [HumanMessage(content=message)], "user_id": session_id},
                {"configurable": {"thread_id": session_id}},
            )
            output_state = await graph.aget_state(
                config={"configurable": {"thread_id": session_id}}
            )

        response_message = output_state.values["messages"][-1].content
        workflow = output_state.values.get("workflow", "conversation")

        return JSONResponse(
            content={
                "reply": response_message,
                "session_id": session_id,
                "route_used": workflow,
            },
            status_code=200
        )

    except KeyError as e:
        logger.error(f"Missing field in state: {e}", exc_info=True)
        return JSONResponse(content={"error": "No response generated"}, status_code=500)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@chat_router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)