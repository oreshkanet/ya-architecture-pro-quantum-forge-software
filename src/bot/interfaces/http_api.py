"""
HTTP API — FastAPI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from core.bot import RAGBot
from core.config import CHROMA_HOST, CHROMA_PORT


app = FastAPI(title="Wiki RAG Bot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = RAGBot(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)


class AskRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    query: str
    filters: Dict[str, Any]
    answer: str
    chunks: list
    timing: dict


@app.post("/ask", response_model=AskResponse)
async def ask_api(req: AskRequest):
    try:
        result = bot.ask(req.query, filters=req.filters or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "bot_ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)