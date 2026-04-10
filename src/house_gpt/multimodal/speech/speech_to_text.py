# house_gpt/multimodal/speech.py
import asyncio
import os
from house_gpt.core.exceptions import SpeechToTextError
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings
from groq import AsyncGroq

logger = AppLogger("SpeechToText")

SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "webm", "mp4", "mpeg"}


class SpeechToText:
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY) 
        self.model = settings.STT_MODEL_NAME
        logger.info(f"[STT] Initialized with model={self.model}")

    async def transcribe(self, audio_data: bytes, filename: str = "audio.wav") -> str:
        if not audio_data:
            raise SpeechToTextError("Audio data cannot be empty")

        logger.info(f"[STT] Transcribing {len(audio_data)} bytes")

        try:
            transcription = await asyncio.wait_for(
                self._call_groq(audio_data, filename),
                timeout=30
            )

            if not transcription:
                raise SpeechToTextError("Transcription result is empty")

            logger.info(f"[STT] Transcription done: {transcription[:80]!r}")
            return transcription

        except asyncio.TimeoutError:
            logger.error("[STT] Transcription timed out after 30s")
            raise SpeechToTextError("Transcription timed out. Try a shorter recording.")
        except SpeechToTextError:
            raise
        except Exception as e:
            logger.error(f"[STT] Transcription failed: {e}", exc_info=True)
            raise SpeechToTextError(f"Transcription failed: {str(e)}") from e

    async def _call_groq(self, audio_data: bytes, filename: str) -> str:
        response = await self.client.audio.transcriptions.create(
            file=(filename, audio_data),
            model=self.model,
            language="en",
            response_format="text",
        )
        return response.strip() if isinstance(response, str) else response.text.strip()