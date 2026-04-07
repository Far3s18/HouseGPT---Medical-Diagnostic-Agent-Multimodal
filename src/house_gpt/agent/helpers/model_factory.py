from house_gpt.core.settings import settings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def get_small_model(temperature: float = 0.7):
    return ChatOllama(
        model=settings.SMALL_TEXT_MODEL_NAME,
        temperature=temperature,
        max_retries=2
    )

def get_large_model(temperature: float = 0.7):
    return ChatOllama(
        model=settings.LARGE_TEXT_MODEL_NAME,
        temperature=temperature,
        max_retries=2
    )