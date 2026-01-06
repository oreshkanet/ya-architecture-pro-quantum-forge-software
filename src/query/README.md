# Query

Устанавливаем зависимости:
```sh
pip3 install -r ./src/query/requirements.txt
```

Запускаем поиск:
```sh
SCHEMA_PATH="./volumes/index_schema.json" python3 ./src/query/query.py "Кто такой Voy?"
```