#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилита для анализа логов запросов к RAG-боту.

Позволяет получить статистику по запросам из query_analytics.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import Dict, Any, List

# Добавляем путь к src/bot для импорта модулей
project_root = Path(__file__).parent.parent
bot_path = project_root / "bot"
sys.path.insert(0, str(bot_path))

from core.query_logger import QueryLogger


def analyze_logs(
    log_file: str,
    days: int = 7,
    top_n: int = 10,
    show_unanswered: bool = True,
) -> None:
    """
    Анализирует логи и выводит статистику.
    
    Args:
        log_file: Путь к файлу логов.
        days: Количество дней для анализа.
        top_n: Количество топ-запросов для вывода.
        show_unanswered: Показывать ли неотвеченные запросы.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Файл логов не найден: {log_path}")
        sys.exit(1)
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    entries = []
    unanswered_queries = []
    all_queries = []
    
    print(f"📖 Чтение логов из: {log_path}")
    print(f"📅 Анализ запросов за последние {days} дней\n")
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time < cutoff_date:
                        continue
                    
                    entries.append(entry)
                    all_queries.append(entry["query"])
                    
                    if not entry.get("chunks_found") or not entry.get("is_successful"):
                        unanswered_queries.append({
                            "query": entry["query"],
                            "timestamp": entry["timestamp"],
                            "chunks_found": entry.get("chunks_found", False),
                            "is_successful": entry.get("is_successful", False),
                        })
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга строки {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        sys.exit(1)
    
    if not entries:
        print(f"📭 Нет записей за последние {days} дней")
        return
    
    total_queries = len(entries)
    successful = sum(1 for e in entries if e.get("is_successful"))
    with_chunks = sum(1 for e in entries if e.get("chunks_found"))
    avg_answer_length = sum(e.get("answer_length", 0) for e in entries) / total_queries
    
    print("="*70)
    print("📊 СТАТИСТИКА ЗАПРОСОВ")
    print("="*70)
    print(f"\n📈 Общая статистика:")
    print(f"   Всего запросов: {total_queries}")
    print(f"   Успешных ответов: {successful} ({successful/total_queries*100:.1f}%)")
    print(f"   Запросов с найденными чанками: {with_chunks} ({with_chunks/total_queries*100:.1f}%)")
    print(f"   Запросов без чанков: {total_queries - with_chunks} ({(total_queries-with_chunks)/total_queries*100:.1f}%)")
    print(f"   Средняя длина ответа: {avg_answer_length:.0f} символов")
    
    # Топ частых запросов
    query_counts = Counter(all_queries)
    print(f"\n🔝 Топ-{top_n} самых частых запросов:")
    for i, (query, count) in enumerate(query_counts.most_common(top_n), 1):
        print(f"   {i}. [{count}x] {query}")
    
    # Неотвеченные запросы
    if show_unanswered and unanswered_queries:
        print(f"\n❓ Неотвеченные запросы ({len(unanswered_queries)}):")
        unique_unanswered = {}
        for uq in unanswered_queries:
            q = uq["query"]
            if q not in unique_unanswered:
                unique_unanswered[q] = 0
            unique_unanswered[q] += 1
        
        for i, (query, count) in enumerate(sorted(unique_unanswered.items(), key=lambda x: -x[1])[:top_n], 1):
            print(f"   {i}. [{count}x] {query}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Анализ логов запросов к RAG-боту"
    )
    parser.add_argument(
        "--log-file",
        "-f",
        default="query_analytics.jsonl",
        help="Путь к файлу логов (по умолчанию: query_analytics.jsonl)",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Количество дней для анализа (по умолчанию: 7)",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=10,
        help="Количество топ-запросов для вывода (по умолчанию: 10)",
    )
    parser.add_argument(
        "--no-unanswered",
        action="store_true",
        help="Не показывать неотвеченные запросы",
    )
    
    args = parser.parse_args()
    
    analyze_logs(
        log_file=args.log_file,
        days=args.days,
        top_n=args.top,
        show_unanswered=not args.no_unanswered,
    )


if __name__ == "__main__":
    main()
