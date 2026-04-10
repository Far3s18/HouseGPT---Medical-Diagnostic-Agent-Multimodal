from house_gpt.core.settings import settings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_small_model(temperature: float = 0.7):
    return ChatOpenAI(
        model=settings.SMALL_TEXT_MODEL_NAME,
        base_url=settings.OPENROUTER_URL,
        api_key=settings.OPENROUTER_API_KEY,
        temperature=temperature,
        max_retries=2
    )

def get_large_model(temperature: float = 0.7):
    return ChatOpenAI(
        model=settings.LARGE_TEXT_MODEL_NAME,
        base_url=settings.OPENROUTER_URL,
        api_key=settings.OPENROUTER_API_KEY,
        temperature=temperature,
        max_retries=2
    )

def get_image_to_text_model():
    return ChatOpenAI(
        model=settings.ITT_MODEL_NAME,
        base_url=settings.OPENROUTER_URL,
        api_key=settings.OPENROUTER_API_KEY,
        max_tokens=1024,
    )