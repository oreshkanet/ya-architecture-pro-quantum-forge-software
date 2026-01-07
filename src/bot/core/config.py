import os

SECURITY_ENABLED = bool(os.getenv("SECURITY_ENABLED", False))

# === Ollama LLM ===
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

# === ChromaDB ===
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# === Retrieval ===
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))
MAX_CONTEXT_LEN = int(os.getenv("MAX_CONTEXT_LEN", "4096"))

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
BOT_HISTORY_PATH = os.getenv("BOT_HISTORY_PATH", "bot_history.jsonl")

# === Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")