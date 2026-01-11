# Тестирование качества базы знаний

## Установка зависимостей

Перед использованием установите необходимые зависимости:

```bash
pip install -r src/tester/requirements.txt
```

## Запуск тестирования

```bash
python3 src/tester/test_golden_questions.py
```

## Запуск тестирования с параметрами

```bash
# Подробный вывод
python3 src/tests/test_golden_questions.py --verbose

# Указать путь к файлу с вопросами
python3 src/tests/test_golden_questions.py --questions knowledge_config/golden_questions.json

# Указать хост и порт ChromaDB
python3 src/tests/test_golden_questions.py --chroma-host localhost --chroma-port 8000

# Указать файл для сохранения результатов
python3 src/tests/test_golden_questions.py --results volumes/tests/test_results.json

# Не сохранять результаты в файл
python3 src/tests/test_golden_questions.py --no-save
```

## Формат результатов

Результаты сохраняются в JSON файл со следующей структурой:

```json
{
  "test_metadata": {
    "timestamp": "2025-01-17T12:00:00Z",
    "total_questions": 15,
    "total_time_seconds": 45.2
  },
  "overall_statistics": {
    "correct_answers": 12,
    "incorrect_answers": 3,
    "accuracy_percent": 80.0,
    "avg_response_time_seconds": 3.01
  },
  "known_topics_statistics": {
    "total": 9,
    "correct": 8,
    "accuracy_percent": 88.89
  },
  "unknown_topics_statistics": {
    "total": 6,
    "correct": 4,
    "accuracy_percent": 66.67
  },
  "results": [
    {
      "question_id": "gq_001",
      "query": "Что такое Orthography?",
      "category": "known",
      "evaluation": {
        "overall_correct": true,
        "chunks_match": true,
        "success_match": true
      }
    }
  ]
}
```
