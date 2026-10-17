"""Agent API."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from scripts.retrieve_tool_info import run_SciToolAgent
from app.core.config import Config
from app.core.logger import get_logger
from app.core.runtime_identity import (
    capture_runtime_identity,
    read_runtime_identity_snapshot,
)
import uvicorn
import time

_cfg = Config()
logger = get_logger("AgentService", "agent.log")
_PROJECT_ROOT = Path(__file__).resolve().parent
_SERVICE_STARTED_AT = datetime.now(timezone.utc).isoformat()
_SERVICE_RUNTIME_IDENTITY = capture_runtime_identity(
    service_name="agent",
    project_root=_PROJECT_ROOT,
    configured_port=_cfg.SERVER_PORT,
    started_at=_SERVICE_STARTED_AT,
    dependencies={"biology_api_base": _cfg.BIO_API_BASE},
)

def create_app():
    app = FastAPI(title="BioAgent API Service", version="1.0")
    
    class ChatQuery(BaseModel):
        question: str
        # Legacy client fields are accepted for compatibility; configuration is server-side.
        api_key: str | None = None
        api_base: str | None = None
        is_add_filename: bool = False
        file_path_list: list

    @app.get("/api/runtime-info")
    def runtime_info():
        try:
            return read_runtime_identity_snapshot(_SERVICE_RUNTIME_IDENTITY)
        except Exception as exc:
            logger.error("读取Agent运行身份失败: %s", exc, exc_info=True)
            return {"status": "error", "message": f"读取Agent运行身份失败: {exc}"}

    @app.post("/chat")
    async def chat(query: ChatQuery):
        start_time = time.time()
        question = query.question
        is_add_filename = query.is_add_filename
        file_path_list = query.file_path_list
        
        logger.info(f"Received chat request | Question: {question} | is_add_filename: {is_add_filename} | Files: {file_path_list}")
        try:
            result = await run_SciToolAgent(question, file_path_list, is_add_filename=is_add_filename)
            latency = time.time() - start_time
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            reasoning = result.get("reasoning", {}) if isinstance(result, dict) else {}
            logger.info(f"Successfully processed chat request in {latency:.2f}s | Answer preview: {answer[:150]}...")
            return {"response": answer, "reasoning": reasoning, "latency": round(latency, 2)}
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Failed to process chat request after {latency:.2f}s | Error: {type(e).__name__}: {str(e)}", exc_info=True)
            return {
                "error_type": type(e).__name__,
                "message": str(e)
            }
    
    return app

app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting BioAgent API Service on port {_cfg.SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=_cfg.SERVER_PORT)
