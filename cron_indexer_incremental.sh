#!/usr/bin/env bash

###############################################################################
# Ежедневный (или периодический) запуск инкрементальной индексации
# indexer --mode incremental --s3-sync через docker-compose.
#
# Как использовать:
#   1) Сделать файл исполняемым:
#        chmod +x /opt/rag/cron_indexer_incremental.sh
#   2) Добавить в crontab (пример — запуск каждый день в 02:00):
#        0 2 * * * /opt/rag/cron_indexer_incremental.sh >> /var/log/rag_indexer_cron.log 2>&1
#
# Важно:
#   - код проекта на сервере располагается в папке /opt/rag
#   - предполагается, что docker и docker-compose доступны в PATH
###############################################################################

set -euo pipefail

PROJECT_DIR="/opt/rag"
cd "$PROJECT_DIR"

# Опционально: можно явно указать путь к docker-compose, если cron не видит PATH
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-docker-compose}"

TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[$TIMESTAMP] Starting incremental index update (docker-compose run --rm indexer --mode incremental --s3-sync)"

# Запуск инкрементальной индексации
$DOCKER_COMPOSE_BIN run --rm indexer --mode incremental --s3-sync

EXIT_CODE=$?
END_TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[$END_TIMESTAMP] Incremental index update finished successfully (exit code 0)"
else
  echo "[$END_TIMESTAMP] Incremental index update finished with errors (exit code $EXIT_CODE)"
fi

exit "$EXIT_CODE"

