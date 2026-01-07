import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", choices=["cli", "http", "telegram"], required=True)
    parser.add_argument("--query", help="Вопрос (только для CLI)")
    args, remaining = parser.parse_known_args()

    if args.interface == "cli":
        from interfaces import cli
        import sys
        sys.argv = ["cli.py"] + ([args.query] if args.query else []) + remaining
        cli.run()
    elif args.interface == "http":
        from interfaces import http_api
        import uvicorn
        uvicorn.run("interfaces.http_api:app", host="0.0.0.0", port=8080, reload=True)
    elif args.interface == "telegram":
        from interfaces import telegram_bot
        asyncio.run(telegram_bot.run_telegram())

if __name__ == "__main__":
    main()