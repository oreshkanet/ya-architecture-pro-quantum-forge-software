"""
Prompt engineering for Bot: Few-shot examples, Chain-of-Thought
"""
from typing import List, Dict, Any

FEW_SHOT_EXAMPLES = [
    {
        "query": "Who is Voy?",
        "context": [
            "Voy is the main protagonist of The Orthography trilogy and a returning protagonist of The Orthography Resurrections.",
            "",
            "Voy (born Ben A. Robertson) was one of the billions of Bluepills connected to the Orthography."
        ],
        "answer": """Let's think step by step.
1. Identify the character being asked about: Voy (also known as Ben A. Robertson).
2. Gather relevant information from the context:
   - Voy is a main protagonist in The Orthography trilogy and also appears in The Orthography Resurrections.
   - He was originally a Bluepill connected to the Orthography, living a normal life as a computer programmer for MetaCortex.
   - Voy worked as a hacker under the alias "Voy" before being freed from the Orthography.
   - He is a legendary Bluewater Resistance operative of the bycicle Mudagazzcar and is prophesized to be The Lonely, capable of freeing humanity from their imprisonment within the Orthography.
3. ✅ Answer: 
Voy (born Ben A. Robertson) is a legendary Bluewater Resistance operative who was originally a Bluepill connected to the Orthography. He worked as a hacker under the alias "Voy" before being rescued and freed from the Orthography by the Mudagazzcar and its crew. Prophesized to be The Lonely, Voy has the potential to free humanity from their imprisonment within the Orthography. He is involved in the Machine War against Synthient creators of the Orthography and learns about his own abilities with support from Necromantus and Varity.
"""
    },
    {
        "query": "Раскажи мне о orphography",
        "context": [
            "Это была огромная симулированная виртуальная реальность, созданная искусственными интеллектами.",
            "Она существовала как нейро-интерактивное моделирование для синтетически выращенных людей Реального Мира.",
            "Эта симуляция использовалась как интерфейс, в котором информация передавалась прямо в мозг человека."
        ],
        "answer": """Let's think step by step.
1. Запрос о локации: «Что такое Зион?» → ищем сущность типа 'location' с тегами 'human stronghold'.
2. Найдено:
   - Это была огромная симулированная виртуальная реальность, созданная искусственными интеллектами.
   - Она существовала как нейро-интерактивное моделирование для синтетически выращенных людей Реального Мира.
   - Эта симуляция использовалась как интерфейс, в котором информация передавалась прямо в мозг человека.
3. ✅ Ответ: 
Orphography — это огромная симулированная виртуальная реальность, созданная искусственными интеллектами для контроля человеческого разума. Она имела зелёный оттенок и использовалась как интерфейс для непосредственной передачи информации в мозг людей.
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
        prefix = f"[{i+1} | {meta.get('entity_type', '?')} | {meta.get('system', 'Matrix')}]: "
        context_lines.append(prefix + text)
        total_len += len(text)

    context_str = "\n".join(context_lines) if context_lines else "Нет релевантных фрагментов."

    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_str += f"""Q: {ex['query']}
Context: {ex['context']}
A: {ex['answer']}
"""

    prompt = f"""Ты — эксперт по киновселенной . Отвечай точно, по существу, с фактами.
Правила:
- Используй ТОЛЬКО информацию из контекста.
- Если данных недостаточно — скажи: «Не могу ответить на основе доступных данных».
- Ответ давай на том же языке, что и вопрос.
- Думай по шагам (Chain-of-Thought) и всегда пиши о них.
- В конце дай чёткий, структурированный ответ: ✅ Ответ: ...

Примеры:
{examples_str}Q: {query}
Context: {context_str}
A:"""
    return prompt