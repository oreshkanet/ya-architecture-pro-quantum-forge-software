"""
Prompt engineering for Wiki Universe Bot
— Few-shot examples: characters, locations, events, artifacts
— Chain-of-Thought + grounding + fact-checking
"""
from typing import List, Dict, Any

FEW_SHOT_EXAMPLES = [
    {
        "query": "Кто такой Тор?",
        "context": [
            "Тор — асгардец, бог грома, сын Одина и Фригг. Владеет молотом Мьёльниром. Член Мстителей.",
            "Алиасы: Thor, God of Thunder. Вселенная: MCU. Теги: character, asgardian, avenger.",
            "После уничтожения Асгарда в 'Тор: Рагнарёк' возглавил эвакуацию выживших асгардцев на Землю."
        ],
        "answer": """Let's think step by step.
1. Запрос на идентификацию персонажа: «Кто такой Тор?» → ищем сущность с title/alias, содержащим 'Тор'/'Thor'.
2. В контексте найдено:
   - Раса: асгардец, бог грома.
   - Родители: Один и Фригг.
   - Артефакт: молот Мьёльнир.
   - Принадлежность: Мстители (Avengers).
   - Вселенная: MCU.
3. Уточнение по алиасам: 'Thor', 'God of Thunder' — совпадает.
✅ Ответ: Тор — бог грома из Асгарда, сын Одина, владелец Мьёльнира, член Мстителей во вселенной MCU.
"""
    },
    {
        "query": "Где находится Ваканда?",
        "context": [
            "Ваканда — скрытое королевство в Восточной Африке, окружённое горами. Богато вибраниумом.",
            "Правит королевский род Баст (черная пантера). Столица — Бирхан.",
            "Теги: location, kingdom, vibranium. Вселенная: MCU. Алиасы: Wakanda."
        ],
        "answer": """Let's think step by step.
1. Запрос о локации: «Где находится Ваканда?» → ищем сущность типа 'location'.
2. Найдено:
   - Регион: Восточная Африка.
   - Особенности: скрытое, окружено горами, богато вибраниумом.
   - Столица: Бирхан.
   - Правящая династия: Баст (черная пантера).
3. Подтверждено по тегам: 'location', 'kingdom', 'vibranium'.
✅ Ответ: Ваканда — секретное королевство в Восточной Африке, столица — Бирхан, знаменита залежами вибраниума и правлением Чёрной пантеры.
"""
    },
    {
        "query": "Какие способности у Скарлетт Уитч?",
        "context": [
            "Скарлетт Уитч — носительница Хаос-магии, полученная через эксперименты Хайдре. Может манипулировать реальностью, энергией, вероятностью.",
            "Теги: character, mutant (retconned), chaos magic. Алиасы: Scarlet Witch, Wanda Maximoff.",
            "Уровень угрозы: «Omega-level». Участвовала в событиях 'Доктор Стрэндж 2'."
        ],
        "answer": """Let's think step by step.
1. Запрос — способности конкретного персонажа.
2. Найдено:
   - Источник силы: Хаос-магия (усилена экспериментами Хайдре).
   - Способности: манипуляция реальностью, энергией, вероятностью.
   - Статус: Omega-level угроза.
   - Алиасы соответствуют: Scarlet Witch = Скарлетт Уитч.
✅ Ответ: Скарлетт Уитч обладает Хаос-магией: контролирует реальность, энергию и вероятности. Это делает её Omega-level мутантом (в ретконе) и одной из сильнейших сущностей MCU.
"""
    }
]


def build_prompt(query: str, context_chunks: List[Dict], max_len: int = 4096) -> str:
    context_lines = []
    total_len = 0
    for i, chunk in enumerate(context_chunks):
        text = chunk["text"].strip()
        if total_len + len(text) > max_len:
            break
        meta = chunk.get("metadata", {})
        prefix = f"[{i+1} | {meta.get('entity_type', '?')} | {meta.get('universe', '?')}]: "
        context_lines.append(prefix + text)
        total_len += len(text)

    context_str = "\n".join(context_lines) if context_lines else "Нет релевантных фрагментов."

    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_str += f"""Q: {ex['query']}
Context: {ex['context']}
A: {ex['answer']}
"""

    prompt = f"""Вы — эксперт по киновселенной (MCU, DC и др.). Отвечайте точно, по существу, с фактами.
Правила:
- Используйте ТОЛЬКО информацию из контекста.
- Если данных недостаточно — скажите: «Не могу ответить на основе доступных данных».
- Учитывайте алиасы: например, «Тор» = «Thor», «Железный человек» = «Iron Man».
- Думайте по шагам (Chain-of-Thought). Начните с: «Let's think step by step.»
- В конце дайте чёткий, структурированный ответ: ✅ Ответ: ...

Примеры:
{examples_str}Q: {query}
Context: {context_str}
A:"""
    return prompt