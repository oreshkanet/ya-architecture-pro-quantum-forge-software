# ya-architecture-pro-quantum-forge-software
Проектная работа 7 спринта курса "Архитектура ПО: продвинутый уровень"

## Первый запуск (с загрузкой моделей)

Поднимаем сервер ollama в docker и загружаем модель
```sh
docker-compose up -d ollama
docker-compose exec ollama ollama pull qwen2.5:7b-instruct
```

Запускаем chroma с бэкендом clickhouse:
```sh
docker-compose up -d clickhouse chroma
```

Затем — индексация (запускается вручную):
```sh
docker-compose run --rm indexer
```

Запуск API и бота

HTTP API
```sh
docker-compose up -d rag_http
```

или Telegram
```sh
docker-compose up -d rag_telegram
```

Поиск контекста по векторной базе:
```sh
docker-compose run --rm query "Кто такой Voy?"
```
