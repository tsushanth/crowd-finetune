import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

DB_PATH = os.getenv("DB_PATH", "data/crowd.db")
BASE_LLM_URL = os.getenv("BASE_LLM_URL", "https://api.openai.com/v1")
BASE_LLM_API_KEY = os.getenv("BASE_LLM_API_KEY", "")
BASE_LLM_MODEL = os.getenv("BASE_LLM_MODEL", "qwen2.5-7b-instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_MIN_SCORE = float(os.getenv("JUDGE_MIN_SCORE", "0.9"))
FILTER_THETA = float(os.getenv("FILTER_THETA", "0.8"))
MIN_EVAL_DELTA = float(os.getenv("MIN_EVAL_DELTA", "0.03"))
CORPUS_PATH = Path(
    os.getenv("CORPUS_PATH", str(BASE_DIR / "data" / "corpus.jsonl"))
)
if not CORPUS_PATH.is_absolute():
    CORPUS_PATH = BASE_DIR / CORPUS_PATH

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
REQUIRE_TG_AUTH = os.getenv("REQUIRE_TG_AUTH", "false") == "true"