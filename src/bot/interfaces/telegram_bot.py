"""
Telegram bot — aiogram v3+
"""
import os
import logging
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from core.bot import RAGBot
from core.config import TELEGRAM_TOKEN, CHROMA_HOST, CHROMA_PORT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()
rag_bot = RAGBot(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🎬 Привет! Я — Wiki Bot по киновселенным (MCU, DC и др.).\n\n"
        "Примеры вопросов:\n"
        "• Кто такой Тор?\n"
        "• Какие способности у Скарлетт Уитч?\n"
        "• Где находится Ваканда?\n"
        "• Что такое Тессеракт?\n\n"
        "Просто напишите вопрос — и я отвечу на основе официальной Wiki!"
    )


@router.message()
async def handle_message(message: Message):
    query = message.text.strip()
    if not query:
        return

    try:
        # Простая эвристика фильтров из текста
        filters = {}
        lower = query.lower()
        if " mcu" in lower or "марвел" in lower:
            filters["universe"] = "MCU"
        elif " dc" in lower:
            filters["universe"] = "DC"

        result = rag_bot.ask(query, filters=filters)
        answer = result["answer"]

        # Обрезаем до 4096 символов (лимит Telegram)
        if len(answer) > 4000:
            answer = answer[:3997] + "..."

        await message.answer(answer, parse_mode=None)

    except Exception as e:
        logger.exception("Telegram bot error")
        await message.answer(f"⚠️ Ошибка: {e}")


dp.include_router(router)


async def run_telegram():
    logger.info("🚀 Запуск Telegram бота...")
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не установлен в переменных окружения")

    await dp.start_polling(bot)