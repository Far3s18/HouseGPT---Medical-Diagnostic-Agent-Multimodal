import asyncio
import aiosqlite
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from house_gpt.agent.graph.graph import create_workflow_graph
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings
from house_gpt.core.graph_instance import set_graph, set_pool, get_pool, get_graph
from house_gpt.api.v1.routers.chat import chat_router
from house_gpt.memory.ltm.vector_store import _qdrant_client

logger = AppLogger("API-V1")

app = FastAPI(title="HouseGPT API", version="1.0.0")

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.on_event("startup")
async def startup():
    try:
        pool = AsyncConnectionPool(
            conninfo=settings.POSTGRES_URI,
            min_size=settings.POSTGRES_MIN_CONNECTIONS,
            max_size=settings.POSTGRES_MAX_CONNECTIONS,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "connect_timeout": 10,
            }
        )
        await pool.open()
        set_pool(pool)
        logger.info("[STARTUP] PostgreSQL pool opened OK")
    except Exception as e:
        logger.error(f"[STARTUP] PostgreSQL pool FAILED: {e}", exc_info=True)
        raise

    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("[STARTUP] LangGraph checkpoint tables ready")
    except Exception as e:
        logger.error(f"[STARTUP] Checkpointer setup FAILED: {e}", exc_info=True)
        raise

    try:
        graph = create_workflow_graph().compile(checkpointer=checkpointer)
        set_graph(graph)
        logger.info("[STARTUP] Graph compiled OK")
    except Exception as e:
        logger.error(f"[STARTUP] Graph compile FAILED: {e}", exc_info=True)
        raise

    try:
        _qdrant_client.get_collections()
        logger.info("[STARTUP] Qdrant connection OK")
    except Exception as e:
        logger.error(f"[STARTUP] Qdrant connection FAILED: {e}")

    logger.info("[STARTUP] HouseGPT ready")

@app.on_event("shutdown")
async def shutdown():
    pool = get_pool()
    if pool:
        await pool.close()
        logger.info("[SHUTDOWN] PostgreSQL pool closed")

@app.get("/api/v1/health")
async def health():
    return JSONResponse(content={"status": "ok"})

@app.get("/api/v1/health/detailed")
async def detailed_health():
    checks = {}
    try:
        pool = get_pool()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)}"

    try:
        from house_gpt.memory.ltm.vector_store import _qdrant_client
        _qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)}"

    checks["graph"] = "ok" if get_graph() else "not initialized"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    code = 200 if status == "ok" else 503

    return JSONResponse(content={"status": status, "checks": checks}, status_code=code)


app.include_router(chat_router, prefix="/api/v1")