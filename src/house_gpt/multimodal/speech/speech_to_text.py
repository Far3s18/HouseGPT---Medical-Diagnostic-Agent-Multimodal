import os
import asyncio
import tempfile
import whisper
from house_gpt.core.exceptions import SpeechToTextError
from house_gpt.core.settings import settings

class SpeechToText:
    def __init__(self) -> None:
        self.model = whisper.load_model(settings.STT_MODEL_NAME)

    def _transcribe_sync(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result.get("text", "").strip()

    async def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        temp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                temp_file_path = tmp.name

            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._transcribe_sync, temp_file_path
                )

                if not result:
                    raise SpeechToTextError("Transcription result is empty")

                return result

            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            raise SpeechToTextError(f"Speech-to-Text conversion failed: {str(e)}") from e

