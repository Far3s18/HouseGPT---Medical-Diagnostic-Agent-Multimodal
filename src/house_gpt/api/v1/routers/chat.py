from fastapi import APIRouter, Request, Response, UploadFile, File, Form
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import JSONResponse
from house_gpt.multimodal.speech import SpeechToText
from house_gpt.multimodal.image import ImageToText
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings
from house_gpt.services.graph_services import invoke_graph
from house_gpt.services.image_services import get_image_type
from house_gpt.core.graph_instance import get_graph
import time

logger = AppLogger("API")
chat_router = APIRouter()

stt = SpeechToText()
itt = ImageToText()

MAX_AUDIO_SIZE = 10 * 1024 * 1024
MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_MESSAGE_LENGTH = 2000

ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@chat_router.post("/chat")
async def chat_handler(request: Request):
    try:
        start = time.time()
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "")

        if not message:
            return JSONResponse(content={"error": "Message is required"}, status_code=400)
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return JSONResponse(content={"error": f"Message exceeds {MAX_MESSAGE_LENGTH} characters"}, status_code=400)


        logger.info(
            f"[START] session={session_id} message={message[:50]}"
        )

        output_state = await invoke_graph(message, session_id)

        response_message = output_state.values["messages"][-1].content
        workflow = output_state.values.get("workflow", "conversation")

        end = time.time()
        duration = end - start
        print(f"Duration Taked: {duration}")

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

        if audio.content_type not in ALLOWED_AUDIO_TYPES:
            return JSONResponse(
                content={"error": f"Unsupported audio type: {audio.content_type}"},
                status_code=415
            )

        audio_data = await audio.read()

        if not audio_data:
            return JSONResponse(content={"error": "Audio file is empty"}, status_code=400)

        if len(audio_data) > MAX_AUDIO_SIZE:
            return JSONResponse(content={"error": "Audio file exceeds 10MB limit"}, status_code=413)


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


@chat_router.post("/chat/image")
async def chat_image_handler(session_id: str = Form(...), image: UploadFile = File(...), prompt: str = Form(default="")):
    try:
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)

        content_type = get_image_type(image, ALLOWED_IMAGE_TYPES)

        if content_type not in ALLOWED_IMAGE_TYPES:
            return JSONResponse(
                content={"error": f"Unsupported image type: {content_type}"},
                status_code=415
            )

        image_data = await image.read()

        if not image_data:
            return JSONResponse(content={"error": "Image file is empty"}, status_code=400)

        if len(image_data) > MAX_IMAGE_SIZE:
            return JSONResponse(content={"error": "Image file exceeds 5MB limit"}, status_code=413)

        logger.info(f"Analyzing image for session {session_id} ({len(image_data)} bytes)")

        image_description = await itt.analyze_image(image_data, prompt)

        if not image_description:
            return JSONResponse(content={"error": "Could not analyze image"}, status_code=422)

        
        logger.info(f"Image description: {image_description!r}")

        output_state = await invoke_graph(image_description, session_id)

        return JSONResponse(content={
            "reply": output_state.values["messages"][-1].content,
            "session_id": session_id,
            "route_used": output_state.values.get("workflow", "conversation"),
            "image_description": image_description,
        }, status_code=200)


    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)



@chat_router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)