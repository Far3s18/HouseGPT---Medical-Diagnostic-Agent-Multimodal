import re
from langchain_core.output_parsers import StrOutputParser

def remove_asterisk_content(text: str) -> str:
    return re.sub(r"\*.*?\*", "", text).strip()

class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return remove_asterisk_content(super().parse(text))

def get_format_memories(self, memories: List[str]) -> str:
    if not memories:
        return ""
    return "\n".join(f"- {memory}" for memory in memories)