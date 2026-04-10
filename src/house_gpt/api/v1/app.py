import asyncio
import aiosqlite
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from house_gpt.agent.graph.graph import create_workflow_graph
from fastapi.middleware.cors import CORSMiddleware
from house_gpt.core.logger import AppLogger
from house_gpt.core.settings import settings
from house_gpt.core.graph_instance import set_graph, set_pool, get_pool, get_graph
from house_gpt.api.v1.routers.chat import chat_router
from house_gpt.memory.ltm.vector_store import _qdrant_client
from fastapi import Request

def get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


logger = AppLogger("API-V1")

app = FastAPI(title="HouseGPT API", version="1.0.0")
limiter = Limiter(key_func=get_real_ip)
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
        logger.error("[STARTUP] PostgreSQL pool FAILED", exc_info=True)
        raise e

    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        async with pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_daily_quota (
                    user_id       TEXT    NOT NULL,
                    day           DATE    NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, day)
                )
            """)
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_quota_day
                        ON user_daily_quota (day)
                """)
            except Exception:
                pass

        logger.info("[STARTUP] Quota table ready")
        logger.info("[STARTUP] LangGraph checkpoint tables ready")
    except Exception as e:
        logger.error("[STARTUP] Checkpointer setup FAILED", exc_info=True)
        raise e

    try:
        graph = create_workflow_graph().compile(checkpointer=checkpointer)
        set_graph(graph)
        logger.info("[STARTUP] Graph compiled OK")
    except Exception as e:
        logger.error("[STARTUP] Graph compile FAILED", exc_info=True)
        raise e

    try:
        _qdrant_client.get_collections()
        logger.info("[STARTUP] Qdrant connection OK")
    except Exception:
        logger.error("[STARTUP] Qdrant connection FAILED", exc_info=True)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")