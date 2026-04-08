from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RAG:
    text: str
    book_title: str
    metadata: dict
    score: Optional[float] = None