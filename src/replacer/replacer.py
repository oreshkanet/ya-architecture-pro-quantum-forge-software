import os
import re
import json
from pathlib import Path

SOURCE_DIR = Path(os.getenv("SOURCE_DIR", "source"))
TARGET_DIR = Path(os.getenv("TARGET_DIR", "target"))
MAP_FILE = Path(os.getenv("MAP_FILE", "terms_map.json"))

# Опция: заменять ТОЛЬКО целые слова (не подстроки)
REPLACE_WHOLE_WORDS_ONLY = False

def load_replacement_dict(map_path: Path) -> dict:
    if not map_path.exists():
        raise FileNotFoundError(f"Файл словаря замены не найден: {map_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("terms_map.json должен содержать JSON-объект (словарь).")
    return data


def replace_in_text(text: str, replacements: dict, whole_words: bool = False) -> str:
    for old, new in replacements.items():
        if whole_words:
            pattern = r'\b' + re.escape(old) + r'\b'
            text = re.sub(pattern, new, text)
        else:
            # простая замена подстроки
            text = text.replace(old, new)
    return text


def main():
    replacements = load_replacement_dict(MAP_FILE)
    print(f"Загружено {len(replacements)} замен из {MAP_FILE}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    ound_any = False
    for filepath in SOURCE_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        found_any = True

        rel_path = filepath.relative_to(SOURCE_DIR)
        target_file = TARGET_DIR / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            text = filepath.read_text(encoding="utf-8")
            new_text = replace_in_text(text, replacements, REPLACE_WHOLE_WORDS_ONLY)
            target_file.write_text(new_text, encoding="utf-8")
            print(f"✅ {rel_path}")
        except UnicodeDecodeError:
            print(f"⚠️ Пропущен (не текстовый или кодировка не UTF-8): {rel_path}")
        except Exception as e:
            print(f"❌ Ошибка при обработке {rel_path}: {e}")

    if not found_any:
        print(f"⚠️ Нет файлов в папке {SOURCE_DIR}")


if __name__ == "__main__":
    main()