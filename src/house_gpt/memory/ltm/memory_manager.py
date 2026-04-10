import uuid
from house_gpt.core.logger import AppLogger
from functools import lru_cache
from house_gpt.agent.helpers.model_factory import get_small_model
from house_gpt.states.memory import MemoryAnalysis
from house_gpt.agent.prompts import MEMORY_ANALYSIS_PROMPT
from house_gpt.core.settings import settings
from .vector_store import get_vector_store
from langchain_core.messages import BaseMessage
from datetime import datetime
from typing import List


class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_store = get_vector_store(self.user_id)
        self.logger = AppLogger(name="Memory Manager")
        self.llm = get_small_model(temperature=0.1).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        if message.type != "human":
            return

        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def get_relevant_memories(self, context: str) -> List[str]:
        memories = self.vector_store.search_memories(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}'", score=f"{memory.score:.2f}")
        return [memory.text for memory in memories]


@lru_cache(maxsize=256)
def get_memory_manager(user_id: str) -> MemoryManager:
    return MemoryManager(user_id)