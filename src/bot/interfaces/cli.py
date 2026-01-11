"""
CLI interface for Wiki RAG Bot
"""
import argparse
from core.bot import RAGBot
from core.config import CHROMA_HOST, CHROMA_PORT


def run():
    parser = argparse.ArgumentParser(description="Wiki RAG Bot — CLI")
    parser.add_argument("query", help="Вопрос по киновселенной")
    parser.add_argument("--universe", help="Фильтр: MCU, DC, StarWars и т.д.")
    parser.add_argument("--entity-type", help="character, location, artifact, event, faction...")
    parser.add_argument("--tags", nargs="*", help="Фильтр по тегам (через пробел)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод метаданных")
    args = parser.parse_args()

    filters = {}
    if args.universe:
        filters["universe"] = args.universe
    if args.entity_type:
        filters["entity_type"] = args.entity_type
    if args.tags:
        filters["tags"] = args.tags

    bot = RAGBot(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)
    result = bot.ask(query=args.query, filters=filters)

    print(f"\n🎬 Wiki Bot — вопрос: «{args.query}»")
    print("=" * 60)
    if result["chunks"]:
        print(f"✅ Найдено {len(result['chunks'])} релевантных фрагментов")
        if args.verbose:
            for i, c in enumerate(result["chunks"]):
                meta = c["metadata"]
                title = meta.get("title", "?")
                etype = meta.get("entity_type", "?")
                uni = meta.get("universe", "?")
                print(f"  [{i+1}] {title} [{etype}] <{uni}> | score={c['score']:.4f}")
    else:
        print("⚠️ Ничего не найдено — генерация без контекста.")

    print("\n" + "=" * 60)
    print("💡 Ответ:")
    print()
    print(result["answer"])
    print("\n" + "=" * 60)

    timing = result["timing"]
    print(f"⏱️ Время: поиск {timing['retrieve']:.2f}с + генерация {timing['generate']:.2f}с = {timing['total']:.2f}с")