"""
Main RAG Bot class — для Wiki киновселенной
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional
from .config import LOG_LEVEL, TOP_K_CONTEXT, MAX_CONTEXT_LEN
from .retrieval import retrieve_context
from .prompting import build_prompt
from .llm_client import OllamaClient


class RAGBot:
    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        top_k: int = TOP_K_CONTEXT,
        max_context_len: int = MAX_CONTEXT_LEN,
        log_history_to: Optional[str] = "bot_history.jsonl"
    ):
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.top_k = top_k
        self.max_context_len = max_context_len
        self.log_history_to = log_history_to
        self.client = OllamaClient()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    def ask(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start = time.time()

        # 1. Retrieve
        chunks = retrieve_context(
            query=query,
            top_k=self.top_k,
            filters=filters,
            chroma_host=self.chroma_host,
            chroma_port=self.chroma_port
        )
        retrieve_time = time.time() - start

        # 2. Build prompt
        prompt = build_prompt(query, chunks, max_len=self.max_context_len)

        # 3. Generate
        gen_start = time.time()
        answer = self.client.generate(prompt)
        gen_time = time.time() - gen_start

        result = {
            "query": query,
            "filters": filters or {},
            "chunks": chunks,
            "answer": answer,
            "timing": {
                "retrieve": retrieve_time,
                "generate": gen_time,
                "total": retrieve_time + gen_time
            }
        }

        # 4. Log history
        if self.log_history_to:
            self._log_to_file(result)

        return result

    def _log_to_file(self, result: Dict[str, Any]):
        with open(self.log_history_to, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")