"""
RAG Retrieval — reuse query_engine hybrid search
"""
from typing import List, Dict, Any, Optional
from tools.query_engine import QueryEngine


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
        chunks.append({
            "id": res["id"],
            "text": res.get("document") or "",
            "score": res.get("rerank_score", res.get("score_rrf", 0.0)),
            "metadata": res["metadata"],
        })
    return chunks