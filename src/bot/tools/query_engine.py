#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Query Engine — для Wiki киновселенной.
Поддерживает гибридный поиск, rerank, фильтрацию по типам сущностей/вселенным/тегам,
логирование запросов и разрешение имён через алиасы.
"""

import os
import sys
import argparse
import logging
import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from chromadb import HttpClient
from chromadb.config import Settings

# Конфигурация по умолчанию
DEFAULT_CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
DEFAULT_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "index_schema.json"))
COLLECTION_KB_DENSE = "knowledge_base_dense"
COLLECTION_KB_SPARSE = "knowledge_base_sparse"
COLLECTION_QUERY_LOG = "query_log"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_RERANKED = 5

logger = logging.getLogger("QueryEngine")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class QueryEngine:
    def __init__(
        self,
        chroma_host: str = DEFAULT_CHROMA_HOST,
        chroma_port: int = DEFAULT_CHROMA_PORT,
        verbose: bool = False,
    ):
        self.verbose = verbose
        setup_logging(verbose)

        # Подключение к Chroma
        try:
            self.client = HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
            self.client.heartbeat()
            logger.info(f"✅ Подключено к ChromaDB ({chroma_host}:{chroma_port})")
        except Exception as e:
            logger.error(f"❌ ChromaDB недоступен: {e}")
            sys.exit(1)

        # Загрузка схемы
        if not SCHEMA_PATH.exists():
            logger.error(f"❌ Схема не найдена: {SCHEMA_PATH}")
            sys.exit(1)
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        logger.info(f"📜 Схема загружена: {self.schema['domain']} — {self.schema['created_at']}")

        self.dense_collection = self.client.get_collection(COLLECTION_KB_DENSE)
        self.sparse_collection = self.client.get_collection(COLLECTION_KB_SPARSE)

        # Загрузка моделей
        logger.info("Загрузка эмбеддера и reranker...")
        self.embedder = BGEM3FlagModel(
            model_name_or_path=EMBEDDING_MODEL,
            use_fp16=(DEVICE == "cuda"),
            device=DEVICE,
        )
        self.reranker = FlagReranker(
            model_name_or_path=RERANKER_MODEL,
            use_fp16=(DEVICE == "cuda"),
            device=DEVICE,
        )
        logger.info("✅ Модели готовы.")

    def _encode_dense(self, query: str) -> np.ndarray:
        return self.embedder.encode(
            [query],
            batch_size=1,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0]

    def _encode_sparse(self, query: str) -> Dict[int, float]:
        sparse_weights = self.embedder.encode(
            [query],
            batch_size=1,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"][0]

        token_ids = self.embedder.tokenizer.convert_tokens_to_ids(list(sparse_weights.keys()))
        return {
            tid: float(sparse_weights[token])
            for tid, token in zip(token_ids, sparse_weights.keys())
            if tid not in (self.embedder.tokenizer.unk_token_id, self.embedder.tokenizer.pad_token_id)
        }

    def _normalize_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        # Chroma требует {"$eq": ...}, {"$in": [...]}, и т.д.
        chroma_filter = {}
        for k, v in filters.items():
            if k not in self.schema.get("filters_supported", []):
                logger.warning(f"Фильтр '{k}' не поддерживается — игнорируется.")
                continue
            if isinstance(v, str):
                chroma_filter[k] = {"$eq": v}
            elif isinstance(v, (list, tuple)):
                chroma_filter[k] = {"$in": list(v)}
        return chroma_filter or None

    def _resolve_aliases(self, candidates: List[Dict]) -> List[Dict]:
        """
        Улучшает recall: если запрос содержит "Ванесса", а в базе только "Vanessa",
        или есть алиасы — повышаем релевантность таких совпадений.
        Реализуем простое fuzzy-сопоставление по заголовкам и алиасам.
        """
        # Отключаем для reranked — делается до rerank
        return candidates  # placeholder — можно расширить позже

    def search(
        self,
        query: str,
        *,
        top_k: int = TOP_K_RERANKED,
        search_mode: str = "hybrid_rrf",  # 'dense', 'sparse', 'hybrid_rrf'
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        logger.info(f"🔎 Запрос: '{query}' | mode={search_mode} | filters={filters}")

        # Нормализация фильтров
        chroma_filters = self._normalize_filters(filters)

        # Получаем кандидатов
        dense_results = []
        sparse_results = []

        if search_mode in ("dense", "hybrid_rrf"):
            dense_emb = self._encode_dense(query)
            dense_results = self.dense_collection.query(
                query_embeddings=[dense_emb.tolist()],
                n_results=TOP_K_DENSE,
                where=chroma_filters,
                include=["metadatas", "documents", "distances"] if include_metadata else ["metadatas"],
            )

        if search_mode in ("sparse", "hybrid_rrf"):
            sparse_vec = self._encode_sparse(query)
            # Для sparse — ищем по совпадению токенов, используя sparse_vec в `where_document` нельзя напрямую.
            # Используем `query_documents` + фильтры, или эмулируем BM25-подобный поиск через `query` + `where`.
            # Здесь упрощённо: ищем по тексту + метаданным.
            sparse_results = self.sparse_collection.query(
                query_texts=[query],
                n_results=TOP_K_SPARSE,
                where=chroma_filters,
                include=["metadatas", "documents", "distances"] if include_metadata else ["metadatas"],
            )

        # Объединение результатов (RRF — Reciprocal Rank Fusion)
        all_results = []
        id_to_score = {}
        id_to_doc = {}

        # Dense: чем меньше расстояние — тем выше ранг. Преобразуем в score.
        if dense_results.get("ids") and dense_results["ids"][0]:
            ids = dense_results["ids"][0]
            distances = dense_results["distances"][0]
            docs = dense_results["documents"][0] if include_metadata else [None] * len(ids)
            metas = dense_results["metadatas"][0]
            for rank, (doc_id, dist, meta, doc) in enumerate(zip(ids, distances, metas, docs), start=1):
                score = 1.0 / (rank + 60)  # RRF: k=60 по умолчанию
                id_to_score[doc_id] = id_to_score.get(doc_id, 0.0) + score
                id_to_doc[doc_id] = {"id": doc_id, "metadata": meta, "document": doc, "dense_rank": rank}

        # Sparse: используем расстояние от `query_texts` (косинус или L2 — зависит от индекса)
        if sparse_results.get("ids") and sparse_results["ids"][0]:
            ids = sparse_results["ids"][0]
            distances = sparse_results["distances"][0]
            docs = sparse_results["documents"][0] if include_metadata else [None] * len(ids)
            metas = sparse_results["metadatas"][0]
            for rank, (doc_id, dist, meta, doc) in enumerate(zip(ids, distances, metas, docs), start=1):
                score = 1.0 / (rank + 60)
                id_to_score[doc_id] = id_to_score.get(doc_id, 0.0) + score
                if doc_id in id_to_doc:
                    id_to_doc[doc_id]["sparse_rank"] = rank
                else:
                    id_to_doc[doc_id] = {"id": doc_id, "metadata": meta, "document": doc, "sparse_rank": rank}

        # Сортируем по суммарному RRF-счёту
        combined = sorted(
            ({"score_rrf": id_to_score[doc_id], **id_to_doc[doc_id]} for doc_id in id_to_score),
            key=lambda x: -x["score_rrf"],
        )[:TOP_K_DENSE + TOP_K_SPARSE]

        # Rerank (если есть кандидаты и режим — rerank)
        if combined and search_mode != "sparse":
            pairs = [(query, item["document"]) for item in combined if item.get("document")]
            if pairs:
                scores = self.reranker.compute_score(pairs)
                if isinstance(scores, float):
                    scores = [scores]
                for item, score in zip(combined, scores):
                    item["rerank_score"] = float(score)

                combined.sort(key=lambda x: -x.get("rerank_score", -1e9))

        # Оставляем топ-K
        rerank_threshold = -1.0
        result = [
            item for item in combined
            if item.get("rerank_score", -1e9) >= rerank_threshold
        ][:top_k]

        # Логируем запрос
        try:
            log_col = self.client.get_collection(COLLECTION_QUERY_LOG)
        except:
            log_col = self.client.create_collection(COLLECTION_QUERY_LOG)

        log_entry = {
            "query": query,
            "search_mode": search_mode,
            "filters": json.dumps(filters or {}),
            "top_k_requested": top_k,
            "result_count": len(result),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": hashlib.sha1(str(datetime.now()).encode()).hexdigest()[:12],
        }
        log_col.add(
            ids=[hashlib.sha256(f"{query}::{log_entry['timestamp']}".encode()).hexdigest()[:24]],
            documents=["query_log_entry"],
            metadatas=[log_entry],
        )

        logger.info(f"✅ Найдено {len(result)} результатов (запрошено {top_k})")
        return result

    def lookup_by_name(
        self,
        name: str,
        universe: Optional[str] = None,
        exact: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Прямой поиск по `title`, `canonical_name`, `aliases`.
        Полезно для связывания упоминаний в диалоге: "Кто такой Тор?" → ищем сущность с title/alias="Thor".
        """
        name_norm = name.strip().lower()

        # Сначала пробуем точное совпадение по title/canonical_name
        res = self.dense_collection.get(
            where={
                "$or": [
                    {"title": {"$eq": name}},
                    {"canonical_name": {"$eq": name}},
                ]
            },
            limit=1,
        )
        if res["ids"]:
            return {
                "id": res["ids"][0],
                "metadata": res["metadatas"][0],
                "document": res["documents"][0] if res["documents"] else None,
            }

        # Если не найдено — ищем в aliases (contains)
        # Chroma не поддерживает full-text search по строкам напрямую, если поле не indexed как строка.
        # Поэтому делаем запрос по всем записям и фильтруем локально (малая база — допустимо).
        # Альтернатива: хранить aliases как отдельные документы с тегом `is_alias=True`.

        all_meta = self.dense_collection.get(include=["metadatas", "documents"], limit=10000)  # ⚠️ не для больших БД
        for meta, doc, doc_id in zip(all_meta["metadatas"], all_meta["documents"], all_meta["ids"]):
            aliases = [a.strip().lower() for a in meta.get("aliases", "").split(",") if a.strip()]
            if name_norm in aliases or name_norm == meta.get("title", "").lower():
                if universe and meta.get("universe") != universe:
                    continue
                return {
                    "id": doc_id,
                    "metadata": meta,
                    "document": doc,
                }
        return None


def main():
    parser = argparse.ArgumentParser(description="Wiki Query Engine (киновселенная)")
    parser.add_argument("query", nargs="?", help="Текст запроса")
    parser.add_argument("--mode", "-m", default="hybrid_rrf", choices=["dense", "sparse", "hybrid_rrf"])
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K_RERANKED)
    parser.add_argument("--universe", "-u", help="Фильтр по вселенной")
    parser.add_argument("--entity-type", "-t", help="Фильтр по типу сущности")
    parser.add_argument("--tags", nargs="*", help="Фильтр по тегам (через пробел)")
    parser.add_argument("--lookup", "-l", help="Поиск по имени/алиасу (возвращает 1 совпадение)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    engine = QueryEngine(verbose=args.verbose)

    filters = {}
    if args.universe:
        filters["universe"] = args.universe
    if args.entity_type:
        filters["entity_type"] = args.entity_type
    if args.tags:
        filters["tags"] = args.tags

    if args.lookup:
        res = engine.lookup_by_name(args.lookup, universe=args.universe)
        if res:
            meta = res["metadata"]
            print(f"✅ Найдено: {meta['title']} ({meta['entity_type']})")
            print(f"    Universe: {meta.get('universe')}")
            print(f"    Tags: {meta.get('tags')}")
            print(f"    Aliases: {meta.get('aliases')}")
            if res["document"]:
                snippet = (res["document"][:300] + "...") if len(res["document"]) > 300 else res["document"]
                print(f"\n📄 {snippet}")
        else:
            print(f"❌ Сущность '{args.lookup}' не найдена.")
        return

    if not args.query:
        print("Укажите запрос или --lookup <имя>")
        sys.exit(1)

    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        search_mode=args.mode,
        filters=filters,
    )

    print(f"\n🔍 Результаты для: '{args.query}' ({len(results)}):\n")
    for i, item in enumerate(results, 1):
        meta = item["metadata"]
        title = meta.get("title", "—")
        etype = meta.get("entity_type", "—")
        uni = meta.get("universe", "—")
        score = item.get("rerank_score", item.get("score_rrf", 0.0))
        print(f"{i}. {title} [{etype}] <{uni}> — score: {score:.4f}")
        if item.get("document"):
            snippet = item["document"][:200].replace("\n", " ").strip()
            print(f"   → {snippet}...")
        if meta.get("aliases"):
            print(f"     Алиасы: {meta['aliases']}")
        print()


if __name__ == "__main__":
    main()