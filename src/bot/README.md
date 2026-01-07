# Bot

## CLI
python main.py --interface cli "Кто такой Тор?" --universe MCU

## HTTP API
python main.py --interface http
> http://localhost:8080/ask

## Telegram
TELEGRAM_TOKEN=123456:ABC-DEF123 python main.py --interface telegram
