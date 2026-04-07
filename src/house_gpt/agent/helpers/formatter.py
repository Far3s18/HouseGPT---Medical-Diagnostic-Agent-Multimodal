import re
from langchain_core.output_parsers import StrOutputParser

def remove_asterisk_content(text: str) -> str:
    return re.sub(r"\*.*?\*", "", text).strip()

class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return remove_asterisk_content(super().parse(text))