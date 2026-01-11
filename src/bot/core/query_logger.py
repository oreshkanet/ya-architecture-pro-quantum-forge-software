#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль логирования запросов для аналитики покрытия и качества базы знаний.

Каждый запрос сохраняется с полной информацией:
- текст запроса
- timestamp
- были ли найдены чанки
- длина ответа
- флаг «успешный ответ» (по длине, осмысленности или ключевым словам)
- найденные источники
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class QueryLogger:
    """Логирование запросов к RAG-боту для аналитики качества БЗ."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_to_chroma: bool = False,
        chroma_client=None,
        chroma_collection_name: str = "query_analytics",
    ):
        """
        Args:
            log_file: Путь к JSONL файлу для логирования. Если None, логирование в файл отключено.
            log_to_chroma: Сохранять ли логи в ChromaDB коллекцию.
            chroma_client: Экземпляр ChromaDB клиента (если log_to_chroma=True).
            chroma_collection_name: Имя коллекции в ChromaDB для логов.
        """
        self.log_file = log_file
        self.log_to_chroma = log_to_chroma
        self.chroma_client = chroma_client
        self.chroma_collection_name = chroma_collection_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📝 Логирование запросов в: {log_path}")
        
        if self.log_to_chroma and self.chroma_client:
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    self.chroma_collection_name
                )
                self.logger.info(f"✅ Коллекция ChromaDB '{chroma_collection_name}' готова")
            except Exception as e:
                self.logger.warning(f"⚠️ Коллекция не найдена, создаём: {e}")
                try:
                    self.chroma_collection = self.chroma_client.create_collection(
                        self.chroma_collection_name
                    )
                except Exception as create_error:
                    self.logger.error(f"❌ Не удалось создать коллекцию: {create_error}")
                    self.log_to_chroma = False
    
    def log_query(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        timing: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Логирует запрос с полной аналитикой.
        
        Args:
            query: Текст запроса пользователя.
            answer: Сгенерированный ответ бота.
            chunks: Список найденных чанков (может быть пустым).
            timing: Словарь с временами выполнения (retrieve, generate, total).
            filters: Применённые фильтры (если есть).
            session_id: ID сессии пользователя (опционально).
            
        Returns:
            Словарь с записанной информацией о запросе.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Проверка наличия чанков
        chunks_found = len(chunks) > 0
        
        # Длина ответа
        answer_length = len(answer) if answer else 0
        
        # Определяем успешность ответа
        is_successful = self._evaluate_success(
            answer=answer,
            chunks=chunks,
            query=query,
        )
        
        # Собираем источники (metadata из чанков)
        sources = []
        for chunk in chunks:
            source_info = {
                "id": chunk.get("id", "unknown"),
                "score": chunk.get("score", 0.0),
                "metadata": chunk.get("metadata", {}),
            }
            sources.append(source_info)
        
        # Формируем запись
        log_entry = {
            "query": query,
            "timestamp": timestamp,
            "chunks_found": chunks_found,
            "chunks_count": len(chunks),
            "answer_length": answer_length,
            "is_successful": is_successful,
            "sources": sources,
            "timing": timing,
            "filters": filters or {},
            "session_id": session_id,
        }
        
        # Логируем в файл
        if self.log_file:
            self._log_to_file(log_entry)
        
        # Логируем в ChromaDB
        if self.log_to_chroma and self.chroma_client:
            self._log_to_chroma(log_entry)
        
        return log_entry
    
    def _evaluate_success(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> bool:
        """
        Оценивает, был ли ответ успешным.
        
        Критерии успешности:
        1. Найдены чанки (chunks_found = True)
        2. Ответ не слишком короткий (минимум 20 символов)
        3. Ответ не содержит стандартных фраз "не знаю" / "не нашёл" / "нет информации"
        4. Средний score чанков выше порога (если есть)
        
        Args:
            answer: Текст ответа.
            chunks: Найденные чанки.
            query: Исходный запрос.
            
        Returns:
            True если ответ считается успешным, иначе False.
        """
        # 1. Проверка наличия чанков
        if not chunks:
            return False
        
        # 2. Проверка длины ответа
        if not answer or len(answer.strip()) < 20:
            return False
        
        # 3. Проверка на стандартные фразы "не знаю"
        negative_phrases = [
            "не знаю",
            "не нашёл",
            "нет информации",
            "не могу ответить",
            "не удалось найти",
            "извините, но",
            "к сожалению",
            "не найдено",
            "don't know",
            "not found",
            "no information",
        ]
        answer_lower = answer.lower()
        for phrase in negative_phrases:
            if phrase in answer_lower:
                # Проверяем контекст - если это основное сообщение, то неуспешно
                # Если это часть более длинного ответа - может быть OK
                if answer_lower.count(phrase) == 1 and len(answer) < 100:
                    return False
        
        # 4. Проверка среднего score чанков
        scores = [chunk.get("score", 0.0) for chunk in chunks if "score" in chunk]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Порог для rerank_score обычно > -1.0, для score_rrf > 0.0
            # Используем более мягкий порог
            min_score = min(scores)
            if min_score < -2.0:  # Слишком низкий score
                return False
        
        return True
    
    def _log_to_file(self, log_entry: Dict[str, Any]):
        """Сохраняет запись в JSONL файл."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"❌ Ошибка записи в файл {self.log_file}: {e}")
    
    def _log_to_chroma(self, log_entry: Dict[str, Any]):
        """Сохраняет запись в ChromaDB коллекцию."""
        try:
            # Генерируем уникальный ID для записи
            entry_id = hashlib.sha256(
                f"{log_entry['query']}::{log_entry['timestamp']}".encode()
            ).hexdigest()[:24]
            
            # Формируем metadata для ChromaDB
            metadata = {
                "query": log_entry["query"],
                "timestamp": log_entry["timestamp"],
                "chunks_found": str(log_entry["chunks_found"]),
                "chunks_count": str(log_entry["chunks_count"]),
                "answer_length": str(log_entry["answer_length"]),
                "is_successful": str(log_entry["is_successful"]),
                "retrieve_time": str(log_entry.get("timing", {}).get("retrieve", 0.0)),
                "generate_time": str(log_entry.get("timing", {}).get("generate", 0.0)),
                "total_time": str(log_entry.get("timing", {}).get("total", 0.0)),
                "session_id": log_entry.get("session_id", ""),
            }
            
            # Документ для поиска - запрос + ответ
            document = f"Query: {log_entry['query']}\nAnswer: {log_entry.get('answer', '')}"
            
            self.chroma_collection.add(
                ids=[entry_id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as e:
            self.logger.error(f"❌ Ошибка записи в ChromaDB: {e}")
    
    def get_statistics(
        self,
        log_file: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Анализирует логи и возвращает статистику.
        
        Args:
            log_file: Путь к лог-файлу (если None, использует self.log_file).
            days: Количество дней для анализа (от текущей даты назад).
            
        Returns:
            Словарь со статистикой.
        """
        log_file = log_file or self.log_file
        if not log_file or not Path(log_file).exists():
            return {"error": "Лог-файл не найден"}
        
        from datetime import timedelta
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        total_queries = 0
        successful_queries = 0
        queries_with_chunks = 0
        queries_without_chunks = 0
        total_answer_length = 0
        unanswered_queries = []
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time < cutoff_date:
                            continue
                        
                        total_queries += 1
                        
                        if entry.get("is_successful"):
                            successful_queries += 1
                        
                        if entry.get("chunks_found"):
                            queries_with_chunks += 1
                        else:
                            queries_without_chunks += 1
                            unanswered_queries.append({
                                "query": entry["query"],
                                "timestamp": entry["timestamp"],
                            })
                        
                        total_answer_length += entry.get("answer_length", 0)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Ошибка парсинга строки лога: {e}")
                        continue
            
            avg_answer_length = (
                total_answer_length / total_queries if total_queries > 0 else 0
            )
            success_rate = (
                successful_queries / total_queries * 100 if total_queries > 0 else 0
            )
            
            return {
                "period_days": days,
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate_percent": round(success_rate, 2),
                "queries_with_chunks": queries_with_chunks,
                "queries_without_chunks": queries_without_chunks,
                "avg_answer_length": round(avg_answer_length, 1),
                "unanswered_queries_count": len(unanswered_queries),
                "top_unanswered_queries": unanswered_queries[:10],
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа логов: {e}")
            return {"error": str(e)}
