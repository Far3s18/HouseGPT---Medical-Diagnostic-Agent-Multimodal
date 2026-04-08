from typing import Literal
from pydantic import BaseModel, Field, field_validator

class RouterResponse(BaseModel):
    response_type: Literal["conversation", "image", "audio", "rag"] = Field(
        description=(
            "The routing decision for the user's message. Must be exactly one of:\n"
            "- 'conversation': general chat, casual questions, simple health chitchat\n"
            "- 'image': user explicitly requests generating an image or visual\n"
            "- 'audio': user explicitly requests hearing audio or voice\n"
            "- 'rag': deep or specific medical question requiring precise clinical knowledge"
        )
    )

    @field_validator("response_type", mode="before")
    @classmethod
    def normalize(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"conversation", "image", "audio", "rag"}
        if normalized not in allowed:
            return "conversation"
        return normalized