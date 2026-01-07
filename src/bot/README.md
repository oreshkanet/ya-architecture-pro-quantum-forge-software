# Bot

## CLI
python3 main.py --interface cli "Кто такой Voy?" --universe MX

## HTTP API
python3 main.py --interface http
> http://localhost:8080/ask

## Telegram
TELEGRAM_TOKEN=123456:ABC-DEF123 python3 main.py --interface telegram
