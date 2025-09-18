# scripts/clean_and_chunk_book.py

import os
import json
from pathlib import Path
from typing import List

from project_config import settings
from scripts import utils

# === Промпт для очистки ===
CLEAN_PROMPT = """Ты — чистильщик текста для подготовки к озвучке.

ЗАДАЧА: из входного фрагмента книги сделать непрерывный, литературно цельный текст для TTS.

СТРОГО УДАЛИ:
- Номера страниц, бегущие колонтитулы, заголовки/подзаголовки, повторяющиеся на каждой странице.
- Все артефакты некорректного PDF→TXT: одинокие цифры, случайные цифры в середине, остатки разметки, номера сносок, звездочки (*), колонтитулы.
- Переносы строк внутри абзацев: склей строки.
- Слова, разрезанные переносом: «компью- \n тер» → «компьютер».
- Лишние множественные пробелы, табы; нормализуй тире и кавычки.
- Пустые строки более одной — схлопни до одной.

СОХРАНИ:
- Основной литературный текст.
- Пунктуацию и абзацы.
- Цифры как часть содержания (годы, списки автора), но НЕ колонтитулы/сноски.

ВЫВОД: только чистый текст без комментариев и префиксов.
"""


from openai import OpenAI, BadRequestError

def openai_clean_chunk(chunk_text: str) -> str:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан (см. .env)")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        # Без temperature — у некоторых моделей допустимо только дефолтное значение
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLEAN_PROMPT},
                {"role": "user", "content": chunk_text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except BadRequestError:
        # Фолбэк на Responses API
        r = client.responses.create(
            model=settings.OPENAI_MODEL,
            input=chunk_text,
            instructions=CLEAN_PROMPT,
        )
        return r.output_text.strip()
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан (см. .env)")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CLEAN_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
    )
    return resp.choices[0].message.content.strip()


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_chunks(chunks: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, piece in enumerate(chunks, 1):
        fname = out_dir / f"{idx:05d}.txt"
        fname.write_text(piece, encoding="utf-8")


def main():
    input_path = Path(settings.BOOK_PATH)
    out_dir = Path(settings.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = input_path.read_text(encoding="utf-8")
    text = utils.soft_normalize(raw)

    # Разбиваем на чанки для LLM
    chunks = utils.chunk_by_tokens(text, settings.MAX_CONTENT_TOKENS)
    print(f"Исходный текст: {len(text):,} символов")
    print(f"Чанков для очистки: {len(chunks)}")

    cleaned_chunks = []
    for i, ch in enumerate(chunks, 1):
        tokens = utils.count_tokens(ch)
        print(f"[{i}/{len(chunks)}] → {tokens:,} токенов, {len(ch):,} символов")
        cleaned = openai_clean_chunk(ch)
        cleaned_chunks.append(cleaned)

    cleaned_full = "\n\n".join(cleaned_chunks).strip()
    cleaned_path = out_dir / "cleaned_full.txt"
    save_text(cleaned_path, cleaned_full)
    print(f"Готово: {cleaned_path} ({len(cleaned_full):,} символов)")

    # Разбиваем для TTS
    tts_chunks = utils.split_for_tts(cleaned_full, settings.SPEECHKIT_CHUNK_SIZE)
    tts_dir = out_dir / "speechkit_chunks"
    save_chunks(tts_chunks, tts_dir)
    print(f"TTS-кусочки: {len(tts_chunks)} шт. → {tts_dir}")


if __name__ == "__main__":
    main()
