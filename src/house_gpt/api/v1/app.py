from fastapi import FastAPI
from house_gpt.core.settings import settings
from house_gpt.api.v1.routers.chat import chat_router

app = FastAPI(title="HouseGPT API", version="1.0.0")
app.include_router(chat_router, prefix="/api/v1")