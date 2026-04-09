# main.py
import aiosqlite
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from house_gpt.agent.graph.graph import create_workflow_graph
from house_gpt.core.settings import settings
from house_gpt.core.graph_instance import set_graph
from house_gpt.api.v1.routers.chat import chat_router

app = FastAPI(title="HouseGPT API", version="1.0.0")

@app.on_event("startup")
async def startup():
    conn = await aiosqlite.connect(settings.SHORT_TERM_MEMORY_DB_PATH)
    checkpointer = AsyncSqliteSaver(conn)
    graph = create_workflow_graph().compile(checkpointer=checkpointer)
    set_graph(graph)

app.include_router(chat_router, prefix="/api/v1")