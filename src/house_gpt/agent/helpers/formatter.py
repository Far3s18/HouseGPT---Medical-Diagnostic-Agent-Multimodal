import re
from langchain_core.output_parsers import StrOutputParser
from typing import List

def remove_asterisk_content(text: str) -> str:
    return re.sub(r"\*.*?\*", "", text).strip()

class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return remove_asterisk_content(super().parse(text))

def get_format_memories(memories: List[str]) -> str:
    if not memories:
        return ""
    return "\n".join(f"- {memory}" for memory in memories)

def build_rag_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r.book_title if hasattr(r, 'book_title') else 'Unknown'} "
            f"| Relevance: {r.score:.2f}\n{r.text}"
        )
    return "\n\n".join(parts)