#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт автоматического тестирования RAG-бота на золотом наборе вопросов.

Проверяет:
- Наличие чанков для каждого вопроса
- Корректность ответа (по длине, наличию ключевых слов, отсутствию "не знаю")
- Соответствие ожидаемым результатам
- Генерирует отчёт о покрытии базы знаний
"""

import json
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Добавляем путь к src/bot для импорта модулей
project_root = Path(__file__).parent.parent
bot_path = project_root / "bot"
sys.path.insert(0, str(bot_path))

# Импорты с учётом структуры проекта (как в http_api.py)
from core.bot import RAGBot
from core.config import CHROMA_HOST, CHROMA_PORT, QUERY_LOG_PATH
from chromadb import HttpClient
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class GoldenQuestionTester:
    """Тестирование RAG-бота на золотом наборе вопросов."""
    
    def __init__(
        self,
        golden_questions_path: str,
        chroma_host: str = CHROMA_HOST,
        chroma_port: int = CHROMA_PORT,
        log_results: bool = True,
        results_file: Optional[str] = None,
    ):
        """
        Args:
            golden_questions_path: Путь к JSON файлу с золотыми вопросами.
            chroma_host: Хост ChromaDB.
            chroma_port: Порт ChromaDB.
            log_results: Сохранять ли результаты в файл.
            results_file: Путь к файлу для сохранения результатов (если None, генерируется автоматически).
        """
        self.golden_questions_path = Path(golden_questions_path)
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.log_results = log_results
        
        # Загружаем золотые вопросы
        with open(self.golden_questions_path, "r", encoding="utf-8") as f:
            self.golden_data = json.load(f)
        
        self.questions = self.golden_data["questions"]
        
        # Инициализируем бота
        try:
            chroma_client = HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
            chroma_client.heartbeat()
        except Exception as e:
            logger.warning(f"ChromaDB недоступен для логирования: {e}")
            chroma_client = None
        
        self.bot = RAGBot(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            enable_query_logging=True,
            chroma_client=chroma_client,
        )
        
        # Путь к файлу результатов
        if results_file:
            self.results_file = Path(results_file)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.results_file = Path(__file__).parent / f"test_results_{timestamp}.json"
        
        self.results = []
    
    def run_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Запускает все тесты из золотого набора.
        
        Args:
            verbose: Выводить ли подробную информацию в процессе.
            
        Returns:
            Словарь с результатами тестирования.
        """
        logger.info(f"🚀 Запуск тестирования на {len(self.questions)} вопросах")
        
        total_questions = len(self.questions)
        known_questions = [q for q in self.questions if q["category"] == "known"]
        unknown_questions = [q for q in self.questions if q["category"] == "unknown"]
        
        logger.info(f"   - Известные темы: {len(known_questions)}")
        logger.info(f"   - Неизвестные темы: {len(unknown_questions)}")
        
        start_time = time.time()
        
        for i, question_data in enumerate(self.questions, 1):
            question_id = question_data["id"]
            query = question_data["query"]
            category = question_data["category"]
            
            if verbose:
                logger.info(f"\n[{i}/{total_questions}] {question_id}: {query}")
            
            # Задаём вопрос боту
            test_start = time.time()
            try:
                result = self.bot.ask(query, filters=None)
                test_time = time.time() - test_start
                
                # Анализируем результат
                test_result = self._analyze_result(
                    question_data=question_data,
                    bot_result=result,
                    test_time=test_time,
                )
                self.results.append(test_result)
                
                if verbose:
                    self._print_result(test_result)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке вопроса {question_id}: {e}")
                test_result = {
                    "question_id": question_id,
                    "query": query,
                    "category": category,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.results.append(test_result)
        
        total_time = time.time() - start_time
        
        # Формируем итоговый отчёт
        summary = self._generate_summary(total_time)
        
        # Сохраняем результаты
        if self.log_results:
            self._save_results(summary)
        
        return summary
    
    def _analyze_result(
        self,
        question_data: Dict[str, Any],
        bot_result: Dict[str, Any],
        test_time: float,
    ) -> Dict[str, Any]:
        """
        Анализирует результат ответа бота на вопрос.
        
        Args:
            question_data: Данные вопроса из золотого набора.
            bot_result: Результат от бота.
            test_time: Время выполнения запроса.
            
        Returns:
            Словарь с анализом результата.
        """
        query = question_data["query"]
        expected_chunks = question_data["expected_chunks_found"]
        expected_success = question_data["expected_success"]
        keywords = question_data.get("keywords", [])
        
        chunks = bot_result.get("chunks", [])
        answer = bot_result.get("answer", "")
        timing = bot_result.get("timing", {})
        
        # Фактические результаты
        chunks_found = len(chunks) > 0
        chunks_count = len(chunks)
        answer_length = len(answer) if answer else 0
        
        # Проверка наличия ключевых слов в ответе (для известных вопросов)
        keywords_found = []
        keywords_missing = []
        if keywords and expected_success:
            answer_lower = answer.lower()
            for keyword in keywords:
                if keyword.lower() in answer_lower:
                    keywords_found.append(keyword)
                else:
                    keywords_missing.append(keyword)
        
        # Оценка успешности ответа
        is_successful = self._evaluate_answer_success(answer, chunks, query)
        
        # Проверка соответствия ожиданиям
        chunks_match = (chunks_found == expected_chunks)
        success_match = (is_successful == expected_success)
        overall_correct = chunks_match and success_match
        
        # Формируем источники
        sources = []
        for chunk in chunks:
            sources.append({
                "id": chunk.get("id", "unknown"),
                "score": chunk.get("score", 0.0),
                "metadata": chunk.get("metadata", {}),
            })
        
        return {
            "question_id": question_data["id"],
            "query": query,
            "category": question_data["category"],
            "description": question_data.get("description", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_time_seconds": round(test_time, 3),
            "expected": {
                "chunks_found": expected_chunks,
                "success": expected_success,
            },
            "actual": {
                "chunks_found": chunks_found,
                "chunks_count": chunks_count,
                "answer_length": answer_length,
                "is_successful": is_successful,
            },
            "evaluation": {
                "chunks_match": chunks_match,
                "success_match": success_match,
                "overall_correct": overall_correct,
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "keywords_match_percent": (
                    len(keywords_found) / len(keywords) * 100
                    if keywords else None
                ),
            },
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "sources": sources[:3],  # Сохраняем только первые 3 источника
            "timing": timing,
        }
    
    def _evaluate_answer_success(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> bool:
        """
        Оценивает, является ли ответ успешным.
        Использует ту же логику, что и QueryLogger.
        """
        if not chunks:
            return False
        
        if not answer or len(answer.strip()) < 20:
            return False
        
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
                if answer_lower.count(phrase) == 1 and len(answer) < 100:
                    return False
        
        scores = [chunk.get("score", 0.0) for chunk in chunks if "score" in chunk]
        if scores:
            min_score = min(scores)
            if min_score < -2.0:
                return False
        
        return True
    
    def _print_result(self, result: Dict[str, Any]):
        """Выводит результат одного теста."""
        status = "✅" if result["evaluation"]["overall_correct"] else "❌"
        print(f"  {status} {result['question_id']}: {result['query']}")
        print(f"     Ожидалось: chunks={result['expected']['chunks_found']}, success={result['expected']['success']}")
        print(f"     Получено: chunks={result['actual']['chunks_found']}, success={result['actual']['is_successful']}")
        if result['evaluation']['keywords_match_percent'] is not None:
            print(f"     Ключевые слова: {result['evaluation']['keywords_match_percent']:.1f}% найдено")
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Генерирует итоговый отчёт о тестировании."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("evaluation", {}).get("overall_correct", False))
        
        known_results = [r for r in self.results if r.get("category") == "known"]
        unknown_results = [r for r in self.results if r.get("category") == "unknown"]
        
        known_correct = sum(1 for r in known_results if r.get("evaluation", {}).get("overall_correct", False))
        unknown_correct = sum(1 for r in unknown_results if r.get("evaluation", {}).get("overall_correct", False))
        
        # Статистика по чанкам
        questions_with_chunks = sum(1 for r in self.results if r.get("actual", {}).get("chunks_found", False))
        avg_chunks = sum(r.get("actual", {}).get("chunks_count", 0) for r in self.results) / total if total > 0 else 0
        avg_answer_length = sum(r.get("actual", {}).get("answer_length", 0) for r in self.results) / total if total > 0 else 0
        
        # Среднее время ответа
        avg_response_time = sum(r.get("test_time_seconds", 0) for r in self.results) / total if total > 0 else 0
        
        summary = {
            "test_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "golden_questions_file": str(self.golden_questions_path),
                "total_questions": total,
                "total_time_seconds": round(total_time, 2),
            },
            "overall_statistics": {
                "correct_answers": correct,
                "incorrect_answers": total - correct,
                "accuracy_percent": round(correct / total * 100, 2) if total > 0 else 0,
                "questions_with_chunks": questions_with_chunks,
                "questions_without_chunks": total - questions_with_chunks,
                "avg_chunks_per_query": round(avg_chunks, 2),
                "avg_answer_length": round(avg_answer_length, 1),
                "avg_response_time_seconds": round(avg_response_time, 3),
            },
            "known_topics_statistics": {
                "total": len(known_results),
                "correct": known_correct,
                "incorrect": len(known_results) - known_correct,
                "accuracy_percent": round(known_correct / len(known_results) * 100, 2) if known_results else 0,
            },
            "unknown_topics_statistics": {
                "total": len(unknown_results),
                "correct": unknown_correct,
                "incorrect": len(unknown_results) - unknown_correct,
                "accuracy_percent": round(unknown_correct / len(unknown_results) * 100, 2) if unknown_results else 0,
            },
            "results": self.results,
        }
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Сохраняет результаты в JSON файл."""
        try:
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Результаты сохранены в: {self.results_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    def print_summary(self, summary: Dict[str, Any]):
        """Выводит итоговый отчёт в консоль."""
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ О ТЕСТИРОВАНИИ")
        print("="*70)
        
        overall = summary["overall_statistics"]
        known = summary["known_topics_statistics"]
        unknown = summary["unknown_topics_statistics"]
        
        print(f"\n📈 Общая статистика:")
        print(f"   Всего вопросов: {summary['test_metadata']['total_questions']}")
        print(f"   Правильных ответов: {overall['correct_answers']}/{overall['correct_answers'] + overall['incorrect_answers']}")
        print(f"   Точность: {overall['accuracy_percent']}%")
        print(f"   Среднее время ответа: {overall['avg_response_time_seconds']} сек")
        print(f"   Средняя длина ответа: {overall['avg_answer_length']:.0f} символов")
        print(f"   Среднее количество чанков: {overall['avg_chunks_per_query']:.1f}")
        
        print(f"\n✅ Известные темы:")
        print(f"   Всего: {known['total']}")
        print(f"   Правильных: {known['correct']}")
        print(f"   Точность: {known['accuracy_percent']}%")
        
        print(f"\n❌ Неизвестные темы:")
        print(f"   Всего: {unknown['total']}")
        print(f"   Правильных (должны НЕ ответить): {unknown['correct']}")
        print(f"   Точность: {unknown['accuracy_percent']}%")
        
        # Топ ошибок
        incorrect = [r for r in summary["results"] if not r.get("evaluation", {}).get("overall_correct", False)]
        if incorrect:
            print(f"\n⚠️ Вопросы с ошибками ({len(incorrect)}):")
            for result in incorrect[:5]:  # Показываем первые 5
                print(f"   • {result['question_id']}: {result['query']}")
                eval_data = result.get("evaluation", {})
                issues = []
                if not eval_data.get("chunks_match"):
                    issues.append("несоответствие чанков")
                if not eval_data.get("success_match"):
                    issues.append("несоответствие успешности")
                print(f"     Проблемы: {', '.join(issues) if issues else 'не определено'}")
        
        print("\n" + "="*70)
        print(f"📄 Полный отчёт сохранён в: {self.results_file}")
        print("="*70 + "\n")


def setup_logging(verbose: bool = False):
    """Настройка логирования."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование RAG-бота на золотом наборе вопросов"
    )
    parser.add_argument(
        "--questions",
        "-q",
        default="tests/golden_questions.json",
        help="Путь к файлу с золотыми вопросами (по умолчанию: tests/golden_questions.json)",
    )
    parser.add_argument(
        "--chroma-host",
        default=CHROMA_HOST,
        help=f"Хост ChromaDB (по умолчанию: {CHROMA_HOST})",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=CHROMA_PORT,
        help=f"Порт ChromaDB (по умолчанию: {CHROMA_PORT})",
    )
    parser.add_argument(
        "--results",
        "-r",
        help="Путь к файлу для сохранения результатов (по умолчанию генерируется автоматически)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результаты в файл",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Проверяем существование файла с вопросами
    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error(f"❌ Файл с вопросами не найден: {questions_path}")
        sys.exit(1)
    
    # Запускаем тестирование
    tester = GoldenQuestionTester(
        golden_questions_path=str(questions_path),
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        log_results=not args.no_save,
        results_file=args.results,
    )
    
    summary = tester.run_tests(verbose=args.verbose)
    tester.print_summary(summary)
    
    # Возвращаем код выхода в зависимости от результата
    overall_accuracy = summary["overall_statistics"]["accuracy_percent"]
    if overall_accuracy >= 80:
        sys.exit(0)  # Успех
    elif overall_accuracy >= 60:
        sys.exit(1)  # Предупреждение
    else:
        sys.exit(2)  # Ошибка


if __name__ == "__main__":
    main()
