#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Index Builder — это RAG-индексатор, ориентированный 
на точность поиска: гибридные эмбеддинги, reranker, метаданные, иерархический чанкинг.

Поддерживает два режима работы:
1. full — первичное построение индекса с очисткой всех чанков
2. incremental — обновление и дополнение индекса новыми/изменёнными документами

Поддержка источников данных:
- Локальная файловая система
- S3-совместимое хранилище (AWS S3, Yandex Object Storage и др.)
"""

# Стандартные библиотеки — для работы с файлами, логами, датой/временем, типизацией и CLI.
import os
import sys
import argparse
import logging
import hashlib
import json
import re
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

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
DEFAULT_STATE_FILE = Path(os.getenv("STATE_FILE", "./index_state.json"))
DEFAULT_CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
DEFAULT_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_KB_DENSE = "knowledge_base_dense"
COLLECTION_KB_SPARSE = "knowledge_base_sparse"
COLLECTION_QUERY_LOG = "query_log"
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "index_schema.json"))

# S3 конфигурация
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", None)  # Для Yandex Object Storage: https://storage.yandexcloud.net
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", None)
S3_PREFIX = os.getenv("S3_PREFIX", "")  # Префикс для фильтрации файлов в бакете
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", None)
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", None)
S3_REGION = os.getenv("S3_REGION", "us-east-1") # Для Yandex Object Storage: ru-central1

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

# ==================== S3 Support ====================

def sync_from_s3(bucket_name: str, prefix: str, local_dir: Path, 
                 endpoint_url: Optional[str] = None,
                 access_key_id: Optional[str] = None,
                 secret_access_key: Optional[str] = None,
                 region: str = "us-east-1") -> bool:
    """
    Синхронизирует файлы из S3 в локальную директорию.
    Возвращает True при успехе, False при ошибке.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.error("boto3 не установлен. Установите: pip install boto3")
        return False

    try:
        # Создаём клиент S3
        s3_config = {}
        if endpoint_url:
            s3_config["endpoint_url"] = endpoint_url
        if access_key_id and secret_access_key:
            s3_config["aws_access_key_id"] = access_key_id
            s3_config["aws_secret_access_key"] = secret_access_key
        if region:
            s3_config["region_name"] = region

        s3_client = boto3.client("s3", **s3_config)

        # Создаём локальную директорию
        local_dir.mkdir(parents=True, exist_ok=True)

        # Список объектов в бакете
        logger.info(f"📥 Синхронизация из S3: s3://{bucket_name}/{prefix}")
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        downloaded_count = 0
        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                # Пропускаем директории
                if key.endswith("/"):
                    continue
                # Фильтруем только поддерживаемые расширения
                if not any(key.lower().endswith(ext) for ext in [".md", ".mdx", ".txt", ".tags"]):
                    continue

                # Сохраняем структуру директорий
                relative_path = key[len(prefix):].lstrip("/")
                local_file = local_dir / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Скачиваем файл
                try:
                    s3_client.download_file(bucket_name, key, str(local_file))
                    downloaded_count += 1
                    logger.debug(f"  ✓ {relative_path}")
                except Exception as e:
                    logger.warning(f"  ✗ Ошибка скачивания {key}: {e}")

        logger.info(f"✅ Скачано {downloaded_count} файлов из S3")
        return True

    except NoCredentialsError:
        logger.error("❌ Не найдены учётные данные для S3. Проверьте переменные окружения.")
        return False
    except ClientError as e:
        logger.error(f"❌ Ошибка S3: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при синхронизации S3: {e}")
        return False


# ==================== File Operations ====================

def compute_file_hash(filepath: Path) -> str:
    """Вычисляет SHA256 хеш файла."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Ошибка вычисления хеша {filepath}: {e}")
        return ""


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


def load_documents_one(file_path: Path) -> List[Document]:
    """Загружает один файл как список документов."""
    try:
        docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        for doc in docs:
            metadata = {
                "source": str(file_path.resolve()),
                "filename": file_path.name,
                "file_type": "wiki",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            tags = load_tags_for_file(file_path)
            if tags:
                metadata["raw_tags"] = ", ".join(sorted(set(tags)))
            fm = doc.metadata.get("frontmatter", {})
            for k in ["title", "universe", "aliases", "canonical_name", "category"]:
                if k in fm:
                    metadata[k] = fm[k]
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        logger.error(f"Ошибка загрузки {file_path}: {e}")
        return []


# ==================== Entity Detection ====================

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


# ==================== Embedding ====================

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


# ==================== Pipeline ====================

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


# ==================== Index Building ====================

def generate_chunk_ids_and_metadata(nodes: List[BaseNode], source: str) -> Tuple[List[str], List[str], List[Dict]]:
    """Генерирует ID, документы и метаданные для чанков."""
    ids, documents, metadatas = [], [], []
    for i, node in enumerate(nodes):
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if len(text) < MIN_CHUNK_LENGTH:
            continue
        meta_norm = normalize_metadata(node)
        if not meta_norm:
            continue
        # Детерминированный ID: source + hash(text)
        chunk_id = hashlib.sha256(f"{source}::{i}::{text}".encode("utf-8")).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta_norm)
    return ids, documents, metadatas


def build_chunks_for_file(file_path: Path, pipeline: IngestionPipeline) -> List[BaseNode]:
    """Создаёт чанки для одного файла."""
    docs = load_documents_one(file_path)
    if not docs:
        return []
    nodes = pipeline.run(documents=docs)
    logger.debug(f"📄 {file_path.name} → {len(nodes)} чанков")
    return nodes


# Запись dense и sparse эмбеддингов в две отдельные коллекции ChromaDB.
def build_chroma_index_full(
    nodes: List[BaseNode],
    chroma_host: str,
    chroma_port: int,
):
    """Полная перестройка индекса: удаляет старые коллекции и создаёт новые."""
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
        chunk_id = hashlib.sha256(f"{source}::{i}::{text}".encode("utf-8")).hexdigest()[:24]

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta_norm)

    if not ids:
        logger.warning("Нет чанков для индексации")
        return

    # Batch-кодирование:
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


def build_chroma_index_incremental(
    input_dir: Path,
    state_file: Path,
    chroma_host: str,
    chroma_port: int,
    no_delete: bool = False,
    dry_run: bool = False,
):
    """Инкрементальное обновление индекса: добавляет новые/изменённые документы, удаляет устаревшие."""
    # Нормализуем входную директорию
    input_dir = Path(input_dir).resolve()
    
    # === 1. Загрузка состояния ===
    state: Dict[str, Dict[str, Any]] = {}
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            # Конвертируем старые абсолютные пути в относительные (для обратной совместимости)
            converted_state = {}
            for key, value in state_data.items():
                try:
                    # Если ключ - абсолютный путь, конвертируем в относительный
                    old_path = Path(key)
                    if old_path.is_absolute():
                        # Пробуем вычислить относительный путь от input_dir
                        try:
                            rel_path = old_path.relative_to(input_dir)
                        except ValueError:
                            # Если не получается (старая директория), используем только имя файла
                            rel_path = Path(old_path.name)
                        new_key = str(rel_path)
                    else:
                        # Уже относительный путь
                        new_key = key
                    converted_state[new_key] = value
                except Exception as e:
                    logger.warning(f"Ошибка обработки ключа состояния {key}: {e}, используем имя файла")
                    # Используем только имя файла как ключ
                    try:
                        filename = Path(key).name
                        converted_state[filename] = value
                    except Exception:
                        # Если совсем не получается, пропускаем
                        logger.error(f"Не удалось обработать ключ {key}, пропускаем")
            state = converted_state
            logger.info(f"Загружено состояние из {state_file} ({len(state)} файлов)")
        except Exception as e:
            logger.error(f"Ошибка чтения {state_file}: {e}")
            sys.exit(1)

    # === 2. Сканирование текущих файлов ===
    current_files: Dict[str, Dict] = {}  # relative_path -> {absolute_path, mtime, hash}
    supported_ext = {".md", ".mdx", ".txt"}
    try:
        for f in input_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_ext:
                try:
                    # Используем относительный путь от input_dir как ключ
                    rel_path = f.relative_to(input_dir)
                    abs_path = f.resolve()
                    mtime = f.stat().st_mtime
                    file_hash = compute_file_hash(f)
                    current_files[str(rel_path)] = {
                        "absolute_path": str(abs_path),
                        "mtime": mtime,
                        "hash": file_hash
                    }
                except Exception as e:
                    logger.warning(f"Пропуск {f}: {e}")
    except Exception as e:
        logger.error(f"Ошибка сканирования {input_dir}: {e}")
        sys.exit(1)

    logger.info(f"Найдено {len(current_files)} документов в {input_dir}")

    # === 3. Определение изменений ===
    new_files = set(current_files.keys()) - set(state.keys())
    deleted_files = set(state.keys()) - set(current_files.keys())
    modified_files = {
        rel_path for rel_path in set(state.keys()) & set(current_files.keys())
        if state[rel_path]["hash"] != current_files[rel_path]["hash"]
    }

    logger.info(f"🆕 Новые: {len(new_files)} | 🔄 Изменённые: {len(modified_files)} | 🗑️ Удалённые: {len(deleted_files)}")

    if not (new_files or modified_files or deleted_files):
        logger.info("✅ Нет изменений. Выход.")
        return

    if dry_run:
        logger.info("=== DRY RUN ===")
        for f in sorted(new_files):
            logger.info(f"+ NEW: {f}")
        for f in sorted(modified_files):
            logger.info(f"~ MOD: {f}")
        for f in sorted(deleted_files):
            logger.info(f"- DEL: {f}")
        return

    # === 4. Подключение к Chroma ===
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

    try:
        kb_dense = client.get_collection(COLLECTION_KB_DENSE)
        kb_sparse = client.get_collection(COLLECTION_KB_SPARSE)
    except NotFoundError:
        logger.error(f"Коллекции {COLLECTION_KB_DENSE}/{COLLECTION_KB_SPARSE} не найдены. Запустите с режимом 'full' сначала.")
        sys.exit(1)

    embedder = BGEM3Embedder(device=DEVICE)
    pipeline = create_pipeline()

    batch_size = 8 if DEVICE == "mps" else 32
    total_added, total_removed = 0, 0

    # === 5. Обработка удалённых файлов ===
    if deleted_files and not no_delete:
        for rel_path in deleted_files:
            try:
                filename = Path(rel_path).name
                # Ищем чанки по filename, затем фильтруем по source
                try:
                    res = kb_dense.get(where={"filename": filename}, include=["metadatas"])
                    filtered_ids = []
                    
                    for i, metadata in enumerate(res.get("metadatas", [])):
                        if not metadata or not isinstance(metadata, dict):
                            continue
                        
                        source_str = metadata.get("source", "")
                        if not source_str:
                            continue
                        
                        source_path = Path(source_str)
                        # Проверяем соответствие: либо относительный путь совпадает,
                        # либо имя файла совпадает (для совместимости со старыми данными)
                        try:
                            # Пробуем вычислить относительный путь от input_dir
                            source_rel = source_path.relative_to(input_dir)
                            if str(source_rel) == rel_path:
                                filtered_ids.append(res["ids"][i])
                        except ValueError:
                            # Если source не содержит input_dir, проверяем по имени файла
                            # Это для совместимости со старыми индексами
                            if source_path.name == filename:
                                filtered_ids.append(res["ids"][i])
                    
                    if filtered_ids:
                        kb_dense.delete(ids=filtered_ids)
                        kb_sparse.delete(ids=filtered_ids)
                        total_removed += len(filtered_ids)
                        logger.info(f"🗑️ Удалено {len(filtered_ids)} чанков для {rel_path}")
                except Exception as e:
                    logger.warning(f"Ошибка поиска чанков для {rel_path}: {e}")
            except Exception as e:
                logger.error(f"Ошибка удаления {rel_path}: {e}")

    # === 6. Обработка новых и изменённых файлов ===
    for rel_path in sorted(new_files | modified_files):
        try:
            # Получаем абсолютный путь к файлу
            abs_path = current_files[rel_path]["absolute_path"]
            file_path = Path(abs_path)
            filename = file_path.name
            logger.info(f"🔄 Обработка: {rel_path}")

            # Удаляем старые чанки (если файл изменён)
            old_ids = []
            if rel_path in state:
                try:
                    # Ищем старые чанки по filename, затем фильтруем по source
                    res = kb_dense.get(where={"filename": filename}, include=["metadatas"])
                    
                    for i, metadata in enumerate(res.get("metadatas", [])):
                        if not metadata or not isinstance(metadata, dict):
                            continue
                        
                        source_str = metadata.get("source", "")
                        if not source_str:
                            continue
                        
                        source_path = Path(source_str)
                        # Проверяем соответствие по относительному пути
                        try:
                            source_rel = source_path.relative_to(input_dir)
                            if str(source_rel) == rel_path:
                                old_ids.append(res["ids"][i])
                        except ValueError:
                            # Если source не содержит input_dir, проверяем по имени файла
                            if source_path.name == filename:
                                old_ids.append(res["ids"][i])
                except Exception as e:
                    logger.warning(f"Не удалось получить старые ID для {rel_path}: {e}")

            # Генерируем новые чанки
            nodes = build_chunks_for_file(file_path, pipeline)
            if not nodes:
                logger.warning(f"⚠️ Нет чанков для {file_path}")
                continue

            # Используем абсолютный путь для генерации ID и сохранения в ChromaDB
            ids, docs, metas = generate_chunk_ids_and_metadata(nodes, abs_path)
            if not ids:
                continue

            # Кодируем эмбеддинги
            dense_embs_all = []
            sparse_dicts_all = []
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                dense, sparse = embedder.encode(batch)
                dense_embs_all.append(dense)
                sparse_dicts_all.extend(sparse)

            dense_embs = np.vstack(dense_embs_all)

            # Обновляем dense
            kb_dense.upsert(
                ids=ids,
                embeddings=dense_embs.tolist(),
                documents=docs,
                metadatas=metas,
            )

            # Обновляем sparse (сохраняем sparse_vec в metadata)
            metas_sparse = []
            for meta, sparse in zip(metas, sparse_dicts_all):
                top_sparse = dict(sorted(sparse.items(), key=lambda x: -abs(x[1]))[:512])
                meta["sparse_vec"] = json.dumps(top_sparse, separators=(",", ":"))
                metas_sparse.append(meta)

            kb_sparse.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas_sparse,
            )

            # Удаляем старые версии (если были)
            if old_ids:
                to_delete = set(old_ids) - set(ids)
                if to_delete:
                    kb_dense.delete(ids=list(to_delete))
                    kb_sparse.delete(ids=list(to_delete))
                    logger.debug(f"🧹 Устаревших чанков удалено: {len(to_delete)}")

            total_added += len(ids)
            logger.info(f"✅ {rel_path} → {len(ids)} чанков")

        except Exception as e:
            logger.error(f"Ошибка обработки {rel_path}: {e}", exc_info=True)

    # === 7. Обновление состояния ===
    # Сохраняем состояние с относительными путями как ключами
    new_state = {}
    for rel_path, info in current_files.items():
        new_state[rel_path] = {
            "mtime": info["mtime"],
            "hash": info["hash"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(new_state, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Состояние сохранено: {state_file}")
    except Exception as e:
        logger.error(f"Не удалось сохранить {state_file}: {e}")

    # === 8. Логирование события в Chroma ===
    try:
        log_col = client.get_or_create_collection(COLLECTION_QUERY_LOG)
        log_col.add(
            ids=[f"update_{int(datetime.now().timestamp())}"],
            documents=["incremental_update"],
            metadatas=[{
                "event": "incremental_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "added": total_added,
                "removed": total_removed,
                "new_files": len(new_files),
                "modified_files": len(modified_files),
                "deleted_files": len(deleted_files),
            }],
        )
    except Exception as e:
        logger.warning(f"Не удалось записать лог в Chroma: {e}")

    logger.info(f"\n🎉 Обновление завершено: +{total_added} чанков, -{total_removed} чанков")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Wiki Universe RAG Indexer — построение и обновление векторного индекса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы работы:
  full         - Первичное построение индекса с очисткой всех чанков (по умолчанию)
  incremental  - Обновление и дополнение индекса новыми/изменёнными документами

Источники данных:
  Локальная файловая система или S3-совместимое хранилище (AWS S3, Yandex Object Storage)

Переменные окружения для S3:
  S3_ENDPOINT_URL       - URL endpoint (для Yandex: https://storage.yandexcloud.net)
  S3_BUCKET_NAME        - Имя бакета
  S3_PREFIX             - Префикс для фильтрации файлов (опционально)
  S3_ACCESS_KEY_ID      - Access key ID
  S3_SECRET_ACCESS_KEY  - Secret access key
  S3_REGION             - Регион (по умолчанию: us-east-1)

Примеры использования:
  # Полная перестройка индекса из локальной папки
  python indexer.py --mode full --input ./knowledge_base

  # Инкрементальное обновление
  python indexer.py --mode incremental --input ./knowledge_base --state ./index_state.json

  # Синхронизация из S3 и полная перестройка
  export S3_BUCKET_NAME=my-bucket
  export S3_ACCESS_KEY_ID=xxx
  export S3_SECRET_ACCESS_KEY=yyy
  python indexer.py --mode full --s3-sync --input ./knowledge_base
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "incremental"],
        default="full",
        help="Режим работы: full (полная перестройка) или incremental (инкрементальное обновление)"
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_DIR, help="Путь к Wiki (MD/MDX)")
    parser.add_argument("--state", "-s", type=Path, default=DEFAULT_STATE_FILE, help="Файл состояния (для incremental режима)")
    parser.add_argument("--chroma-host", default=DEFAULT_CHROMA_HOST, help="Хост ChromaDB")
    parser.add_argument("--chroma-port", type=int, default=DEFAULT_CHROMA_PORT, help="Порт ChromaDB")
    parser.add_argument("--s3-sync", action="store_true", help="Синхронизировать документы из S3 перед индексацией")
    parser.add_argument("--s3-bucket", help="Имя S3 бакета (переопределяет S3_BUCKET_NAME)")
    parser.add_argument("--s3-prefix", help="Префикс в S3 (переопределяет S3_PREFIX)")
    parser.add_argument("--no-delete", action="store_true", help="Не удалять чанки для удалённых файлов (только для incremental)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано (только для incremental)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    mode_str = "полная перестройка" if args.mode == "full" else "инкрементальное обновление"
    logger.info(f"🚀 Запуск индексации (режим: {mode_str}, BGE-m3 hybrid)")

    # Определяем источник данных
    input_dir = args.input
    
    # Синхронизация из S3, если требуется
    if args.s3_sync:
        bucket_name = args.s3_bucket or S3_BUCKET_NAME
        prefix = args.s3_prefix or S3_PREFIX
        
        if not bucket_name:
            logger.error("❌ Не указан S3_BUCKET_NAME или --s3-bucket")
            sys.exit(1)
        
        # Используем временную директорию, если не указана локальная
        if not args.input or args.input == DEFAULT_INPUT_DIR:
            temp_dir = tempfile.mkdtemp(prefix="rag_indexer_")
            input_dir = Path(temp_dir)
            logger.info(f"📁 Используется временная директория: {input_dir}")
        else:
            input_dir = args.input
            input_dir.mkdir(parents=True, exist_ok=True)
        
        success = sync_from_s3(
            bucket_name=bucket_name,
            prefix=prefix,
            local_dir=input_dir,
            endpoint_url=S3_ENDPOINT_URL,
            access_key_id=S3_ACCESS_KEY_ID,
            secret_access_key=S3_SECRET_ACCESS_KEY,
            region=S3_REGION,
        )
        
        if not success:
            logger.error("❌ Ошибка синхронизации из S3")
            sys.exit(1)

    # Проверяем наличие документов
    if not input_dir.exists():
        logger.error(f"❌ Директория не найдена: {input_dir}")
        sys.exit(1)

    # Выполняем индексацию в зависимости от режима
    if args.mode == "full":
        # Полная перестройка индекса
        docs = load_documents(str(input_dir))
    if not docs:
        logger.error("Нет документов для индексации")
            sys.exit(1)

    pipeline = create_pipeline()
    logger.info("🚀 Запуск LlamaIndex ingestion pipeline...")
    nodes = pipeline.run(documents=docs)
    logger.info(f"👉 Создано {len(nodes)} чанков")

        build_chroma_index_full(nodes, args.chroma_host, args.chroma_port)
    logger.info("\n🎉 Индексация завершена.")
        
    else:  # incremental
        # Инкрементальное обновление
        build_chroma_index_incremental(
            input_dir=input_dir,
            state_file=args.state,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            no_delete=args.no_delete,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
