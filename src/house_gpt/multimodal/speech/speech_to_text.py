import asyncio
from groq import AsyncGroq
from house_gpt.core.exceptions import SpeechToTextError
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings

logger = AppLogger("SpeechToText")


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
            response = await self.client.audio.transcriptions.create(
                file=(filename, audio_data),
                model=self.model,
                language="en",
                response_format="text",
            )

            result = response.strip() if isinstance(response, str) else response.text.strip()

            if not result:
                raise SpeechToTextError("Transcription result is empty")

            logger.info(f"[STT] Done: {result[:80]!r}")
            return result

        except SpeechToTextError:
            raise
        except Exception as e:
            logger.error(f"[STT] Failed: {e}", exc_info=True)
            raise SpeechToTextError(f"Transcription failed: {str(e)}") from e