from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from house_gpt.multimodal.speech import SpeechToText
from house_gpt.multimodal.image import ImageToText
from house_gpt.core.logger import AppLogger
from house_gpt.services.graph_services import invoke_graph
from house_gpt.services.image_services import get_image_type
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio
import time

logger = AppLogger("API")
chat_router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

itt = ImageToText()
stt = SpeechToText()

MAX_AUDIO_SIZE = 10 * 1024 * 1024
MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_MESSAGE_LENGTH = 2000
GRAPH_TIMEOUT = 60
AUDIO_TIMEOUT = 30
IMAGE_TIMEOUT = 20

ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg"}


@chat_router.post("/chat")
@limiter.limit("20/minute")
async def chat_handler(request: Request):
    start = time.time()
    session_id = ""
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip()

        if not message:
            return JSONResponse(content={"error": "Message is required"}, status_code=400)
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return JSONResponse(
                content={"error": f"Message too long. Maximum is {MAX_MESSAGE_LENGTH} characters"},
                status_code=400
            )

        logger.info(f"[CHAT] session={session_id} length={len(message)} preview={message[:50]!r}")

        try:
            output_state = await asyncio.wait_for(
                invoke_graph(message, session_id),
                timeout=GRAPH_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"[CHAT][TIMEOUT] session={session_id} after {GRAPH_TIMEOUT}s")
            return JSONResponse(
                content={"error": "Request timed out. Please try again."},
                status_code=504
            )

        response_message = output_state.values["messages"][-1].content
        workflow = output_state.values.get("workflow", "conversation")
        duration = time.time() - start

        logger.info(f"[CHAT][OK] session={session_id} workflow={workflow} duration={duration:.2f}s")

        return JSONResponse(content={
            "reply": response_message,
            "session_id": session_id,
            "route_used": workflow,
            "transcription": None,
        }, status_code=200)

    except KeyError as e:
        logger.error(f"[CHAT][ERROR] session={session_id} missing key: {e}", exc_info=True)
        return JSONResponse(content={"error": "No response was generated. Please try again."}, status_code=500)
    except Exception as e:
        logger.error(f"[CHAT][ERROR] session={session_id} error: {e}", exc_info=True)
        return JSONResponse(content={"error": "Something went wrong. Please try again."}, status_code=500)


@chat_router.post("/chat/audio")
@limiter.limit("10/minute")
async def chat_audio_handler(request: Request, session_id: str = Form(...), audio: UploadFile = File(...)):
    
    start = time.time()
    try:
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)
        if audio.content_type not in ALLOWED_AUDIO_TYPES:
            logger.warning(f"[AUDIO][REJECTED] session={session_id} type={audio.content_type}")
            return JSONResponse(
                content={"error": f"Unsupported audio format: {audio.content_type}. Accepted: wav, mp3, ogg, webm, mp4"},
                status_code=415
            )

        audio_data = await audio.read()

        if not audio_data:
            return JSONResponse(content={"error": "Audio file is empty"}, status_code=400)
        if len(audio_data) > MAX_AUDIO_SIZE:
            logger.warning(f"[AUDIO][REJECTED] session={session_id} size={len(audio_data)} exceeds limit")
            return JSONResponse(content={"error": "Audio file too large. Maximum size is 10MB."}, status_code=413)

        logger.info(f"[AUDIO][START] session={session_id} size={len(audio_data)}bytes")

        try:
            transcription = await asyncio.wait_for(stt.transcribe(audio_data), timeout=AUDIO_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"[AUDIO][TIMEOUT] session={session_id} transcription exceeded {AUDIO_TIMEOUT}s")
            return JSONResponse(content={"error": "Audio transcription timed out. Try a shorter recording."}, status_code=504)

        if not transcription:
            logger.warning(f"[AUDIO][EMPTY] session={session_id} transcription returned empty")
            return JSONResponse(content={"error": "Could not understand the audio. Please speak clearly and try again."}, status_code=422)

        logger.info(f"[AUDIO][TRANSCRIBED] session={session_id} text={transcription[:80]!r}")

        try:
            output_state = await asyncio.wait_for(invoke_graph(transcription, session_id), timeout=GRAPH_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"[AUDIO][TIMEOUT] session={session_id} graph exceeded {GRAPH_TIMEOUT}s")
            return JSONResponse(content={"error": "Request timed out. Please try again."}, status_code=504)

        duration = time.time() - start
        logger.info(f"[AUDIO][OK] session={session_id} duration={duration:.2f}s")

        return JSONResponse(content={
            "reply": output_state.values["messages"][-1].content,
            "session_id": session_id,
            "route_used": output_state.values.get("workflow", "conversation"),
            "transcription": transcription,
        }, status_code=200)

    except Exception as e:
        logger.error(f"[AUDIO][ERROR] session={session_id} error: {e}", exc_info=True)
        return JSONResponse(content={"error": "Failed to process audio. Please try again."}, status_code=500)


@chat_router.post("/chat/image")
@limiter.limit("10/minute")
async def chat_image_handler(
    request: Request,
    session_id: str = Form(...),
    image: UploadFile = File(...),
    prompt: str = Form(default="")
):
    start = time.time()
    try:
        if not session_id:
            return JSONResponse(content={"error": "Session ID is required"}, status_code=400)

        content_type = get_image_type(image, ALLOWED_IMAGE_TYPES)
        if content_type not in ALLOWED_IMAGE_TYPES:
            logger.warning(f"[IMAGE][REJECTED] session={session_id} type={content_type}")
            return JSONResponse(
                content={"error": f"Unsupported image format: {content_type}. Accepted: jpeg, png, webp, gif"},
                status_code=415
            )

        image_data = await image.read()

        if not image_data:
            return JSONResponse(content={"error": "Image file is empty"}, status_code=400)
        if len(image_data) > MAX_IMAGE_SIZE:
            logger.warning(f"[IMAGE][REJECTED] session={session_id} size={len(image_data)} exceeds limit")
            return JSONResponse(content={"error": "Image too large. Maximum size is 5MB."}, status_code=413)

        logger.info(f"[IMAGE][START] session={session_id} size={len(image_data)}bytes prompt={prompt[:50]!r}")

        try:
            image_description = await asyncio.wait_for(
                itt.analyze_image(image_data, prompt),
                timeout=IMAGE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"[IMAGE][TIMEOUT] session={session_id} analysis exceeded {IMAGE_TIMEOUT}s")
            return JSONResponse(content={"error": "Image analysis timed out. Please try again."}, status_code=504)

        if not image_description:
            logger.warning(f"[IMAGE][EMPTY] session={session_id} no description returned")
            return JSONResponse(content={"error": "Could not analyze the image. Please try a different image."}, status_code=422)

        logger.info(f"[IMAGE][ANALYZED] session={session_id} description={image_description[:80]!r}")

        try:
            output_state = await asyncio.wait_for(invoke_graph(image_description, session_id), timeout=GRAPH_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"[IMAGE][TIMEOUT] session={session_id} graph exceeded {GRAPH_TIMEOUT}s")
            return JSONResponse(content={"error": "Request timed out. Please try again."}, status_code=504)

        duration = time.time() - start
        logger.info(f"[IMAGE][OK] session={session_id} duration={duration:.2f}s")

        return JSONResponse(content={
            "reply": output_state.values["messages"][-1].content,
            "session_id": session_id,
            "route_used": output_state.values.get("workflow", "conversation"),
            "image_description": image_description,
        }, status_code=200)

    except Exception as e:
        logger.error(f"[IMAGE][ERROR] session={session_id} error: {e}", exc_info=True)
        return JSONResponse(content={"error": "Failed to process image. Please try again."}, status_code=500)


@chat_router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)