import os
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from house_gpt.core.settings import settings
from house_gpt.core.exceptions import ImageToTextError
from house_gpt.agent.helpers.model_factory import get_image_to_text_model
from house_gpt.core.logger import AppLogger
from typing import Union, Optional

class ImageToText:
    def __init__(self):
        self.model = get_image_to_text_model()
        self.logger = AppLogger("Image-to-Text")

    async def analyze_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        try:
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            if not prompt:
                prompt = "Describe the image objectively. Do not identify people."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            response = await self.model.ainvoke(messages)
            if not response.content:
                raise ImageToTextError("No response received from the vision model")

            description = response.content
            self.logger.info(f"Generated image description: {description}")
            return description

        except Exception as e:
            raise ImageToTextError(f"Failed to analyze image: {str(e)}") from e

