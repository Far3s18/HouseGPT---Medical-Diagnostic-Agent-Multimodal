from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from house_gpt.multimodal.speech import SpeechToText
from house_gpt.multimodal.image import ImageToText
from house_gpt.core.logger import AppLogger
from house_gpt.services.graph_services import invoke_graph, check_and_increment_quota
from house_gpt.services.image_services import get_image_type
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio
import time

def get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

logger = AppLogger("API")
chat_router = APIRouter()
limiter = Limiter(key_func=get_real_ip)

itt = ImageToText()
stt = SpeechToText()

MAX_AUDIO_SIZE  = 10 * 1024 * 1024
MAX_IMAGE_SIZE  = 5  * 1024 * 1024
MAX_MESSAGE_LENGTH = 2000
GRAPH_TIMEOUT   = 60
AUDIO_TIMEOUT   = 30
IMAGE_TIMEOUT   = 20

ALLOWED_AUDIO_TYPES = {"audio/wav","audio/mpeg","audio/ogg","audio/webm","audio/mp4"}
ALLOWED_IMAGE_TYPES = {"image/jpeg","image/png","image/webp","image/gif","image/jpg"}


async def _guard(session_id: str):
    if not session_id:
        return JSONResponse({"error": "Session ID is required"}, status_code=400)
    within_limit = await check_and_increment_quota(session_id)
    if not within_limit:
        logger.warning(f"[QUOTA] session={session_id} daily limit reached")
        return JSONResponse(
            {"error": "Daily limit reached. You can send up to 10 messages per day."},
            status_code=429,
        )
    return None


async def _call_graph(message: str, session_id: str):
    try:
        output_state = await asyncio.wait_for(
            invoke_graph(message, session_id),
            timeout=GRAPH_TIMEOUT,
        )
        return output_state, None
    except RuntimeError as e:
        if "SERVER_BUSY" in str(e):
            logger.warning(f"[BUSY] session={session_id}")
            return None, JSONResponse(
                {"error": "Server is busy. Please retry in a few seconds."},
                status_code=503,
            )
        raise
    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] session={session_id} after {GRAPH_TIMEOUT}s")
        return None, JSONResponse(
            {"error": "Request timed out. Please try again."},
            status_code=504,
        )


@chat_router.post("/chat")
@limiter.limit("20/minute")
async def chat_handler(request: Request):
    start = time.time()
    session_id = ""
    try:
        data = await request.json()
        message    = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip()

        if not message:
            return JSONResponse({"error": "Message is required"}, status_code=400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return JSONResponse(
                {"error": f"Message too long. Maximum is {MAX_MESSAGE_LENGTH} characters"},
                status_code=400,
            )

        err = await _guard(session_id)
        if err:
            return err

        logger.info(f"[CHAT] session={session_id} length={len(message)} preview={message[:50]!r}")

        output_state, err = await _call_graph(message, session_id)
        if err:
            return err

        workflow = output_state.get("workflow", "conversation")
        duration = time.time() - start
        logger.info(f"[CHAT][OK] session={session_id} workflow={workflow} duration={duration:.2f}s")

        return JSONResponse({
            "reply":        output_state["messages"][-1].content,
            "session_id":   session_id,
            "route_used":   workflow,
            "transcription": None,
        }, status_code=200)

    except KeyError as e:
        logger.error(f"[CHAT][ERROR] session={session_id} missing key: {e}", exc_info=True)
        return JSONResponse({"error": "No response generated. Please try again."}, status_code=500)
    except Exception as e:
        logger.error(f"[CHAT][ERROR] session={session_id} error: {e}", exc_info=True)
        return JSONResponse({"error": "Something went wrong."}, status_code=500)


@chat_router.post("/chat/audio")
@limiter.limit("10/minute")
async def chat_audio_handler(
    request: Request, session_id: str = Form(...), audio: UploadFile = File(...)
):
    start = time.time()
    try:
        err = await _guard(session_id)
        if err:
            return err

        if not any(audio.filename.lower().endswith(ext)
                   for ext in [".mp3",".wav",".ogg",".webm",".mp4"]):
            return JSONResponse(
                {"error": f"Unsupported audio format. Accepted: wav, mp3, ogg, webm, mp4"},
                status_code=415,
            )

        audio_data = await audio.read()
        if not audio_data:
            return JSONResponse({"error": "Audio file is empty"}, status_code=400)
        if len(audio_data) > MAX_AUDIO_SIZE:
            return JSONResponse({"error": "Audio too large. Maximum 10 MB."}, status_code=413)

        logger.info(f"[AUDIO][START] session={session_id} size={len(audio_data)}bytes")

        try:
            transcription = await asyncio.wait_for(
                stt.transcribe(audio_data), timeout=AUDIO_TIMEOUT
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                {"error": "Transcription timed out. Try a shorter recording."},
                status_code=504,
            )

        if not transcription:
            return JSONResponse(
                {"error": "Could not understand audio. Please speak clearly."},
                status_code=422,
            )

        logger.info(f"[AUDIO][TRANSCRIBED] session={session_id} text={transcription[:80]!r}")

        output_state, err = await _call_graph(transcription, session_id)
        if err:
            return err

        duration = time.time() - start
        logger.info(f"[AUDIO][OK] session={session_id} duration={duration:.2f}s")

        return JSONResponse({
            "reply":         output_state["messages"][-1].content,
            "session_id":    session_id,
            "route_used":    output_state.get("workflow", "conversation"),
            "transcription": transcription,
        }, status_code=200)

    except Exception as e:
        logger.error(f"[AUDIO][ERROR] session={session_id} error: {e}", exc_info=True)
        return JSONResponse({"error": "Failed to process audio."}, status_code=500)


@chat_router.get("/health")
async def health():
    return JSONResponse({"status": "ok"}, status_code=200)