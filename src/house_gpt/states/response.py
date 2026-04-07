from pydantic import BaseModel, Field

class RouterResponse(BaseModel):
    response_type: str = Field(description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'")