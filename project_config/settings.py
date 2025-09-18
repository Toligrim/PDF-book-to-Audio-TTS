import os
from dotenv import load_dotenv

# Подгрузим .env
load_dotenv()

# Основные настройки
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOOK_PATH = os.getenv("BOOK_PATH", "./data/book.txt")
OUT_DIR = os.getenv("OUT_DIR", "./out")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
MAX_CONTENT_TOKENS = int(os.getenv("MAX_CONTENT_TOKENS", "9500"))

# Настройки для SpeechKit
SPEECHKIT_CHUNK_SIZE = 200
