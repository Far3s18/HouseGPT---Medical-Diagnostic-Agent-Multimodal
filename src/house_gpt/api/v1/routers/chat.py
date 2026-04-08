from fastapi import APIRouter, Request, Response, UploadFile, File, Form
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from house_gpt.agent.graph.graph import create_workflow_graph
from fastapi.responses import JSONResponse
from house_gpt.multimodal.speech import SpeechToText
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings
from house_gpt.services.graph_services import invoke_graph

logger = AppLogger("API")
chat_router = APIRouter()

stt = SpeechToText()


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

        logger.info(
            f"[START] session={session_id} message={message[:50]}"
        )

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
                "transcription": None,
            },
            status_code=200
        )

    except KeyError as e:
        logger.error(f"Missing field in state: {e}", exc_info=True)
        return JSONResponse(content={"error": "No response generated"}, status_code=500)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@chat_router.post("/chat/audio")
async def chat_audio_handler(session_id: str = Form(...), audio: UploadFile = File(...)):
    try:
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)

        allowed_types = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4"}

        if audio.content_type not in allowed_types:
            return JSONResponse(
                content={"error": f"Unsupported audio type: {audio.content_type}"},
                status_code=415
            )

        audio_data = await audio.read()

        if not audio_data:
            return JSONResponse(content={"error": "Audio file is empty"}, status_code=400)

        logger.info(f"Transcribing audio for session {session_id} ({len(audio_data)} bytes)")

        transcription = await stt.transcribe(audio_data)

        if not transcription:
            return JSONResponse(content={"error": "Could not transcribe audio"}, status_code=422)

        logger.info(f"Transcription: {transcription!r}")

        output_state = await invoke_graph(transcription, session_id)

        return JSONResponse(content={
            "reply": output_state.values["messages"][-1].content,
            "session_id": session_id,
            "route_used": output_state.values.get("workflow", "conversation"),
            "transcription": transcription,
        }, status_code=200)

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)



@chat_router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)