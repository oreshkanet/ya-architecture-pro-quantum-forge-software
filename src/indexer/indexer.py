#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Index Builder — это RAG-индексатор, ориентированный 
на точность поиска: гибридные эмбеддинги, reranker, метаданные, иерархический чанкинг.
"""

# Стандартные библиотеки — для работы с файлами, логами, датой/временем, типизацией и CLI.
import os, sys, argparse, logging, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Для работы с эмбеддингами и токенизацией.
import torch
import numpy as np
from transformers import AutoTokenizer

# ChromaDB — векторная БД для хранения dense и sparse эмбеддингов.
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

# LlamaIndex — фреймворк для RAG и ingestion pipeline (чанкинг, извлечение метаданных).
from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.llms import MockLLM

# Модель BGE-M3, поддерживающая одновременно dense, sparse эмбеддинги.
from FlagEmbedding import BGEM3FlagModel

# Отключает предупреждения о параллелизме в transformers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Конфигурация - настройки по умолчанию: пути, модели, размеры чанков, пороги
DEFAULT_INPUT_DIR = Path(os.getenv("INPUT_DIR", "./knowledge_base"))
DEFAULT_CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
DEFAULT_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_KB_DENSE = "knowledge_base_dense"
COLLECTION_KB_SPARSE = "knowledge_base_sparse"
COLLECTION_QUERY_LOG = "query_log"
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "index_schema.json"))

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CHUNK_SIZES = [512, 256]
CHUNK_OVERLAP = 64
MIN_CHUNK_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

logger = logging.getLogger("RAGIndexBuilder")

# Настраивает формат и уровень логирования: INFO по умолчанию, DEBUG — при --verbose
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Загружает теги из привязанных к документам файлов (.tags).
def load_tags_for_file(md_path: Path) -> List[str]:
    tags_path = md_path.with_suffix(".tags")
    if tags_path.exists():
        try:
            return [line.strip() for line in tags_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as e:
            logger.warning(f"Ошибка чтения тегов {tags_path}: {e}")
    return []

# Загружает документы из документов (MD/MDX).
def load_documents(input_dir: str) -> List[Document]:
    input_path = Path(input_dir)
    documents = []

    md_files = list(input_path.rglob("*.md")) + list(input_path.rglob("*.mdx")) + list(input_path.rglob("*.txt"))
    if not md_files:
        logger.warning(f"Нет .md/.mdx файлов в {input_dir}")
        return []

    logger.info(f"Загрузка {len(md_files)} Wiki-документов...")
    for f in md_files:
        try:
            docs = SimpleDirectoryReader(input_files=[str(f)]).load_data()
            for doc in docs:
                # Берём метаданные из frontmatter (LlamaIndex делает это автоматически)
                metadata = {
                    "source": str(f.resolve()),
                    "filename": f.name,
                    "file_type": "wiki",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }

                # Выгружаем теги (если есть .tags)
                tags = load_tags_for_file(f)
                if tags:
                    metadata["raw_tags"] = ", ".join(sorted(set(tags)))

                # Переносим frontmatter-поля, если есть
                fm = doc.metadata.get("frontmatter", {})
                for k in ["title", "universe", "aliases", "canonical_name", "category"]:
                    if k in fm:
                        metadata[k] = fm[k]

                doc.metadata.update(metadata)
                documents.append(doc)
        except Exception as e:
            logger.warning(f"Ошибка при чтении {f}: {e}")

    logger.info(f"✅ Загружено {len(documents)} Wiki-документов.")
    return documents

# Классификация типа сущности
ENTITY_PATTERNS = {
    "character": [
        r"\b(?:character|person|hero|villain|anti[-\s]?hero|protagonist|antagonist)\b",
        r"(?:born|died|species|homeworld|affiliation|powers?|abilities)",
    ],
    "location": [
        r"\b(?:planet|moon|star system|galaxy|city|station|space station|fortress|temple|dimension|realm)\b",
        r"(?:coordinates|gravity|climate|population|inhabitants)",
    ],
    "event": [
        r"\b(?:battle|war|invasion|uprising|treaty|ceremony|disaster|coup|summit|incident)\b",
        r"(?:date|year|duration|outcome|casualties)",
    ],
    "artifact": [
        r"\b(?:artifact|weapon|device|technology|relic|crystal|orb|amulet|scepter|ring|armor|shield)\b",
        r"(?:powers?|activation|creator|origin|destruction)",
    ],
    "faction": [
        r"\b(?:faction|organization|guild|alliance|empire|republic|order|clan|house|syndicate)\b",
        r"(?:leader|headquarters|motto|ideology|members)",
    ],
    "species": [
        r"\b(?:species|race|sentient being|lifeform|humanoid|alien|creature)\b",
        r"(?:homeworld|physiology|lifespan|culture|language)",
    ],
    "vehicle": [
        r"\b(?:ship|starship|vessel|fighter|bomber|drone|mech|transport|hovercraft)\b",
        r"(?:class|manufacturer|armament|speed|crew|capacity)",
    ],
    "concept": [r"\b(?:concept|philosophy|religion|magic|force|energy|law|theory)\b"],
}

# Определяет тип сущности по заголовку или содержимому
def detect_entity_type(text: str, metadata: Dict) -> str:
    # Приоритет: frontmatter.category или universe-specific category
    cat = metadata.get("category") or metadata.get("doc_type")
    if isinstance(cat, str):
        cat = cat.strip().lower()
        for k in ENTITY_PATTERNS:
            if k in cat:
                return k

    # Заголовок (title) — часто содержит подсказку
    title = (metadata.get("title") or "").lower()
    for entity_type, patterns in ENTITY_PATTERNS.items():
        if any(re.search(p, title, re.IGNORECASE) for p in patterns[:1]):  # первая фраза — сильная
            return entity_type

    # Поиск по содержимому (первые 500 символов)
    snippet = text[:500].lower()
    scores = {}
    for entity_type, patterns in ENTITY_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, snippet))
        scores[entity_type] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


# Нормализация метаданных - Приводит метаданные к единому формату, добавляет:
# классификацию, теги (по ключевым словам — дополняет классификатор),
# стандартные поля (source, updated_at, chunk_type и др.).
def normalize_metadata(node: BaseNode) -> Dict[str, Any]:
    meta = node.metadata.copy()
    text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
    if not text:
        return {}

    def safe_str(val, default="unknown"):
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    # Определение типа сущности
    entity_type = detect_entity_type(text, meta)

    # Имя: title > filename (без расширения) > фрагмент текста
    title = safe_str(meta.get("title"))
    if title == "unknown":
        title = Path(meta.get("filename", "")).stem.replace("_", " ").title()

    # Алиасы: из frontmatter.aliases (список или строка)
    aliases_raw = meta.get("aliases", [])
    if isinstance(aliases_raw, str):
        aliases = [a.strip() for a in aliases_raw.split(",")]
    elif isinstance(aliases_raw, list):
        aliases = [str(a).strip() for a in aliases_raw]
    else:
        aliases = []
    aliases = [a for a in aliases if a and a.lower() != title.lower()]

    # Теги: raw_tags + entity_type
    tags = set([entity_type])
    raw_tags = meta.get("raw_tags", "")
    if raw_tags:
        tags.update([t.strip().lower() for t in raw_tags.split(",")])
    
    # Вселенная (если не указана — "default")
    universe = safe_str(meta.get("universe"), "default")

    return {
        "source": safe_str(meta.get("source")),
        "filename": safe_str(meta.get("filename")),
        "title": title,
        "canonical_name": title,
        "aliases": ", ".join(sorted(set(aliases))) if aliases else "",
        "entity_type": entity_type,
        "universe": universe,
        "tags": ", ".join(sorted(tags)),
        "chunk_type": "section" if len(text.splitlines()) > 3 else "snippet",
        "ingested_at": safe_str(meta.get("ingested_at")),
    }

# Обёртка над BGEM3FlagModel, инкапсулирующая загрузку и генерацию dense + sparse эмбеддингов
class BGEM3Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        self.device = device
        logger.info(f"Загрузка BGE-M3 (device={device})...")
        self.model = BGEM3FlagModel(
            model_name_or_path=model_name,
            use_fp16=(device == "cuda"),
            device=device,
        )
        logger.info("✅ BGE-M3 готов (dense + sparse)")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        outputs = self.model.encode(
            texts,
            batch_size=len(texts),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = outputs["dense_vecs"]
        sparse_dicts = []
        for sparse_dict in outputs["lexical_weights"]:
            token_ids = self.model.tokenizer.convert_tokens_to_ids(list(sparse_dict.keys()))
            sparse_dicts.append({
                tid: float(sparse_dict[token])
                for tid, token in zip(token_ids, sparse_dict.keys())
                if tid not in (self.model.tokenizer.unk_token_id, self.model.tokenizer.pad_token_id)
            })
        return dense, sparse_dicts

# Запись dense и sparse эмбеддингов в две отдельные коллекции ChromaDB.
def build_chroma_index(
    nodes: List[BaseNode],
    chroma_host: str,
    chroma_port: int,
):
    try:
        client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )
        client.heartbeat()
        logger.info(f"Подключено к ChromaDB ({chroma_host}:{chroma_port})")
    except Exception as e:
        logger.error(f"ChromaDB недоступен: {e}")
        sys.exit(1)

    # Удаление старых коллекций
    for name in [COLLECTION_KB_DENSE, COLLECTION_KB_SPARSE]:
        try:
            client.delete_collection(name)
            logger.info(f"Очищена коллекция '{name}'")
        except NotFoundError:
            pass

    # Инициализация эмбеддера
    embedder = BGEM3Embedder(device=DEVICE)

    # Подготовка данных
    ids, documents, metadatas = [], [], []

    for i, node in enumerate(nodes):
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if len(text) < MIN_CHUNK_LENGTH:
            continue

        meta_norm = normalize_metadata(node)
        if not meta_norm:
            continue

        # Стабильный ID — source + хеш текста
        source = meta_norm["source"]
        # chunk_id = hashlib.sha256(f"{source}::{text[:128]}".encode()).hexdigest()[:24]
        # chunk_id = hashlib.sha256(f"{source}::{i}::{text[:128]}".encode()).hexdigest()[:24]
        chunk_id = hashlib.sha256(f"{source}::{i}::{text}".encode("utf-8")).hexdigest()[:24]

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta_norm)

    if not ids:
        logger.warning("Нет чанков для индексации")
        return

    # Batch-кодирование:
    # - Размер батча адаптируется под устройство (8 для MPS — меньше памяти).
    # - Dense — как embeddings, sparse — сохраняется в metadatas["sparse_vec"] как JSON-строка (Chroma не хранит sparse напрямую).
    # - Sparse вектор усекается до top-512 (достаточно для поиска, экономит память).
    logger.info(f"Кодирование {len(ids)} чанков...")
    batch_size = 8 if DEVICE == "mps" else 32
    dense_embs_all = []
    sparse_dicts_all = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        dense, sparse = embedder.encode(batch)
        dense_embs_all.append(dense)
        sparse_dicts_all.extend(sparse)
        logger.debug(f"{min(i + batch_size, len(documents))}/{len(documents)}")

    dense_embs = np.vstack(dense_embs_all)

    # Создание коллекций в Chroma для Dense
    kb_dense = client.create_collection(
        name=COLLECTION_KB_DENSE,
        metadata={
            "domain": "wiki_universe",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_type": "dense",
            "dim": 1024,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Запись в Chroma — стандартный dense-поиск
    for i in range(0, len(ids), batch_size):
        kb_dense.add(
            ids=ids[i:i + batch_size],
            embeddings=dense_embs[i:i + batch_size].tolist(),
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    # Создание коллекций в Chroma для Sparse
    kb_sparse = client.create_collection(
        name=COLLECTION_KB_SPARSE,
        metadata={
            "domain": "wiki_universe",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_type": "sparse",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Запись в Chroma - Sparse коллекция (хранит sparse как JSON в metadata + документ)
    metadatas_sparse = []
    for meta, sparse in zip(metadatas, sparse_dicts_all):
        top_sparse = dict(sorted(sparse.items(), key=lambda x: -abs(x[1]))[:512])
        meta["sparse_vec"] = json.dumps(top_sparse, separators=(",", ":"))
        metadatas_sparse.append(meta)

    for i in range(0, len(ids), batch_size):
        kb_sparse.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas_sparse[i:i + batch_size],
        )

    logger.info(f"✅ Dense: {len(ids)} чанков в '{COLLECTION_KB_DENSE}'")
    logger.info(f"✅ Sparse: {len(ids)} чанков в '{COLLECTION_KB_SPARSE}'")

    # Логирование перестройки индекса
    try:
        client.delete_collection(COLLECTION_QUERY_LOG)
    except NotFoundError:
        pass
    log_col = client.create_collection(name=COLLECTION_QUERY_LOG)
    log_col.add(
        ids=["rebuild"],
        documents=["index_rebuilt"],
        metadatas=[{
            "event": "index_rebuild",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(ids),
            "universe": metadatas[0].get("universe", "unknown") if metadatas else "unknown",
        }],
    )

    # Экспорт схемы в index_schema.json — для использования в RAG-рантайме.
    schema = {
        "domain": "wiki_universe",
        "dense_collection": COLLECTION_KB_DENSE,
        "sparse_collection": COLLECTION_KB_SPARSE,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "metadata_keys": list(metadatas[0].keys()) if metadatas else [],
        "filters_supported": ["entity_type", "universe", "tags"],
        "search_modes": ["dense", "sparse", "hybrid_rrf"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logger.info(f"📜 Схема сохранена: {SCHEMA_PATH}")

# Создаёт pipeline LlamaIndex
def create_pipeline() -> IngestionPipeline:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    LlamaSettings.tokenizer = tokenizer

    # Иерархический парсер: сначала крупные секции (~512), потом абзацы (~256)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=CHUNK_SIZES,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Извлекаем заголовки
    title_extractor = TitleExtractor(nodes=5, llm=MockLLM())

    return IngestionPipeline(transformations=[node_parser, title_extractor])

def main():
    parser = argparse.ArgumentParser(description="Wiki Universe RAG Indexer (English)")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_DIR, help="Путь к Wiki (MD/MDX)")
    parser.add_argument("--chroma-host", default=DEFAULT_CHROMA_HOST)
    parser.add_argument("--chroma-port", type=int, default=DEFAULT_CHROMA_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info("🚀 Запуск индексации (BGE-m3 hybrid)")

    docs = load_documents(args.input)
    if not docs:
        logger.error("Нет документов для индексации")
        return

    pipeline = create_pipeline()
    logger.info("🚀 Запуск LlamaIndex ingestion pipeline...")
    nodes = pipeline.run(documents=docs)
    logger.info(f"👉 Создано {len(nodes)} чанков")

    build_chroma_index(nodes, args.chroma_host, args.chroma_port)
    logger.info("\n🎉 Индексация завершена.")

if __name__ == "__main__":
    main()