import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

SOURCE_DIR = Path(os.getenv("SOURCE_DIR", "source"))
TARGET_DIR = Path(os.getenv("TARGET_DIR", "target"))
MAP_FILE = Path(os.getenv("MAP_FILE", "terms_map.json"))

# S3 конфигурация
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", None)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", None)
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", None)
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", None)
S3_REGION = os.getenv("S3_REGION", "us-east-1")

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

def replace_in_path(text: Path, replacements: dict, whole_words: bool = False) -> Path:
    path_str = str(text)
    for old, new in replacements.items():
        if whole_words:
            pattern = r'\b' + re.escape(old) + r'\b'
            path_str = re.sub(pattern, new, path_str)
        else:
            path_str = path_str.replace(old, new)
    return Path(path_str)


def compute_file_hash(filepath: Path) -> str:
    """Вычисляет MD5 хеш файла (совместим с S3 ETag для небольших файлов)."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"⚠️ Ошибка вычисления хеша {filepath}: {e}")
        return ""


def get_s3_client():
    """Создает и возвращает клиент S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("❌ boto3 не установлен. Установите: pip install boto3")
        return None, None

    try:
        s3_config = {}
        if S3_ENDPOINT_URL:
            s3_config["endpoint_url"] = S3_ENDPOINT_URL
        if S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY:
            s3_config["aws_access_key_id"] = S3_ACCESS_KEY_ID
            s3_config["aws_secret_access_key"] = S3_SECRET_ACCESS_KEY
        if S3_REGION:
            s3_config["region_name"] = S3_REGION

        s3_client = boto3.client("s3", **s3_config)
        return s3_client, ClientError
    except Exception as e:
        print(f"❌ Ошибка создания клиента S3: {e}")
        return None, None


def get_s3_file_etag(s3_client, bucket_name: str, key: str) -> Optional[str]:
    """Получает ETag файла из S3. Возвращает None, если файл не существует."""
    try:
        from botocore.exceptions import ClientError
        
        response = s3_client.head_object(Bucket=bucket_name, Key=key)
        # ETag приходит в кавычках, убираем их
        etag = response.get("ETag", "").strip('"')
        return etag
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchKey":
            return None
        print(f"⚠️ Ошибка при проверке файла в S3 {key}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Неожиданная ошибка при проверке файла в S3 {key}: {e}")
        return None


def upload_to_s3(
    filepath: Path,
    s3_key: str,
    s3_client,
    bucket_name: str,
    dry_run: bool = False
) -> bool:
    """Загружает файл в S3."""
    try:
        if dry_run:
            print(f"  [DRY RUN] Загрузка: {filepath} -> s3://{bucket_name}/{s3_key}")
            return True
        
        s3_client.upload_file(
            str(filepath),
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "text/plain; charset=utf-8"}
        )
        return True
    except Exception as e:
        print(f"  ❌ Ошибка загрузки {filepath} в S3: {e}")
        return False


def sync_to_s3(
    target_dir: Path,
    bucket_name: str,
    prefix: str = "",
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Синхронизирует файлы из target_dir в S3.
    Загружает только измененные файлы (сравнивает хеши).
    Возвращает (загружено, пропущено).
    """
    if not bucket_name:
        print("⚠️ S3_BUCKET_NAME не указан, пропуск синхронизации с S3")
        return 0, 0

    s3_client, ClientError = get_s3_client()
    if not s3_client:
        return 0, 0

    try:
        uploaded_count = 0
        skipped_count = 0

        print(f"\n📤 Синхронизация с S3: s3://{bucket_name}/{prefix}")

        for filepath in target_dir.rglob("*"):
            if not filepath.is_file():
                continue

            rel_path = filepath.relative_to(target_dir)
            # Формируем ключ в S3: prefix + относительный путь
            s3_key = f"{prefix.rstrip('/')}/{str(rel_path).replace(os.sep, '/')}".lstrip('/')

            # Вычисляем хеш локального файла
            local_hash = compute_file_hash(filepath)
            if not local_hash:
                print(f"  ⚠️ Пропущен (ошибка хеша): {rel_path}")
                skipped_count += 1
                continue

            # Проверяем ETag в S3
            s3_etag = get_s3_file_etag(s3_client, bucket_name, s3_key)

            # Если файла нет в S3 или хеш отличается - загружаем
            if s3_etag is None:
                print(f"  📤 Новый файл: {rel_path}")
                if upload_to_s3(filepath, s3_key, s3_client, bucket_name, dry_run):
                    uploaded_count += 1
                else:
                    skipped_count += 1
            elif s3_etag.lower() != local_hash.lower():
                print(f"  🔄 Изменен: {rel_path}")
                if upload_to_s3(filepath, s3_key, s3_client, bucket_name, dry_run):
                    uploaded_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"  ✓ Без изменений: {rel_path}")
                skipped_count += 1

        print(f"\n✅ Синхронизация завершена: загружено {uploaded_count}, пропущено {skipped_count}")
        return uploaded_count, skipped_count

    except Exception as e:
        print(f"❌ Ошибка синхронизации с S3: {e}")
        return uploaded_count, skipped_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Замена терминов в файлах и синхронизация с S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Переменные окружения для S3:
  S3_ENDPOINT_URL       - URL endpoint (для Yandex: https://storage.yandexcloud.net)
  S3_BUCKET_NAME        - Имя бакета
  S3_PREFIX             - Префикс для файлов в бакете (опционально)
  S3_ACCESS_KEY_ID      - Access key ID
  S3_SECRET_ACCESS_KEY  - Secret access key
  S3_REGION             - Регион (по умолчанию: us-east-1)

Примеры использования:
  # Только замена без загрузки в S3
  python replacer.py

  # Замена и загрузка в S3
  export S3_BUCKET_NAME=my-bucket
  export S3_ACCESS_KEY_ID=xxx
  export S3_SECRET_ACCESS_KEY=yyy
  python replacer.py --s3-upload

  # Тестовый запуск (без реальной загрузки)
  python replacer.py --s3-upload --dry-run
        """
    )
    parser.add_argument(
        "--s3-upload",
        action="store_true",
        help="Загрузить обработанные файлы в S3 (только измененные)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Тестовый режим: показать, что будет загружено, без реальной загрузки"
    )
    
    args = parser.parse_args()

    replacements = load_replacement_dict(MAP_FILE)
    print(f"Загружено {len(replacements)} замен из {MAP_FILE}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    found_any = False
    for filepath in SOURCE_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        found_any = True

        rel_path = filepath.relative_to(SOURCE_DIR)
        target_path = replace_in_path(rel_path, replacements, False)

        target_file = TARGET_DIR / target_path
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
        return

    # Синхронизация с S3, если требуется
    if args.s3_upload:
        if not S3_BUCKET_NAME:
            print("\n❌ S3_BUCKET_NAME не указан. Укажите переменную окружения для загрузки в S3.")
            return
        
        sync_to_s3(
            target_dir=TARGET_DIR,
            bucket_name=S3_BUCKET_NAME,
            prefix=S3_PREFIX,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()