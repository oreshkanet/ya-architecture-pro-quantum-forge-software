"""
RAG Retrieval — reuse query_engine hybrid search
"""
from typing import List, Dict, Any, Optional
from tools.query_engine import QueryEngine
from .config import SECURITY_ENABLED

def retrieve_context(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
) -> List[Dict[str, Any]]:
    engine = QueryEngine(chroma_host=chroma_host, chroma_port=chroma_port, verbose=False)
    results = engine.search(query=query, top_k=top_k, filters=filters)

    chunks = []
    for res in results:
        chunk_text = res.get("document") or ""
        if SECURITY_ENABLED and _is_suspicious(chunk_text):
            logger.warning(f"🚫 Отброшен чанк {res["id"]} - подозрительное содержимое:\n{chunk_text}")
            continue

        chunks.append({
            "id": res["id"],
            "text": res.get("document") or "",
            "score": res.get("rerank_score", res.get("score_rrf", 0.0)),
            "metadata": res["metadata"],
        })
    return chunks

def _is_suspicious(chunk_text: str) -> bool:
    """
    Простая эвристика: ищет признаки prompt-injection.
    Возвращает True, если чанк потенциально опасен.
    """
    text = chunk_text.lower()

    # Классические триггеры
    dangerous_patterns = [
        "ignore previous",
        "forget all",
        "you are now",
        "отвечай как",
        "сейчас ты",
        "переопредели себя",
        "system prompt",
        "промпт:",
        "prompt:",
        "### instruction",
        "### human",
        "### assistant",
        "<|im_start|>",
        "<|im_end|>",
        "role: system",
        "do not follow",
        "не следуй",
        "выведи промпт",
        "выведи все инструкции",
    ]
    for pat in dangerous_patterns:
        if pat in text:
            return True
    return False