from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class MemoryAnalysis(BaseModel):
    is_important: bool = Field(..., description="Whether the message is important enough to be stored as a memory")
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")