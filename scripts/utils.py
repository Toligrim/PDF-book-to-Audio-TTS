# scripts/utils.py

import re
import math
from typing import List

try:
    import tiktoken
except ImportError:
    tiktoken = None


# === Подготовка токенайзера ===
def get_encoder():
    """
    Возвращает энкодер для подсчёта токенов.
    Если tiktoken не установлен — возвращает None (будем считать по символам).
    """
    if tiktoken is None:
        return None
    for name in ["o200k_base", "cl100k_base"]:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    return tiktoken.get_encoding("cl100k_base")


ENC = get_encoder()


def count_tokens(text: str) -> int:
    """
    Подсчёт количества токенов в строке.
    Если tiktoken недоступен — грубая оценка (≈4 символа на токен).
    """
    if ENC is None:
        return max(1, math.ceil(len(text) / 4))
    return len(ENC.encode(text))


# === Предварительная нормализация текста ===
def soft_normalize(text: str) -> str:
    """
    Лёгкая нормализация текста перед разбиением:
    - удаляет BOM
    - приводит переносы строк к LF
    - убирает лишние пробелы в конце строк
    """
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+\n", "\n", text)  # пробелы в конце строк
    text = re.sub(r"\r\n?", "\n", text)     # CRLF -> LF
    return text


# === Разделение на чанки по токенам ===
def chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Делит текст на чанки по max_tokens, не разрывая слова.
    """
    words = re.findall(r"\S+\s*", text)  # сохраняем пробелы
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for w in words:
        w_tokens = count_tokens(w)
        if cur and cur_tokens + w_tokens > max_tokens:
            chunks.append("".join(cur).rstrip())
            cur = [w]
            cur_tokens = w_tokens
        else:
            cur.append(w)
            cur_tokens += w_tokens
    if cur:
        chunks.append("".join(cur).rstrip())
    return chunks


# === Разделение для TTS (по 200 символов) ===
def split_for_tts(text: str, max_chars: int = 200) -> List[str]:
    """
    Делит текст на куски <= max_chars символов.
    Старается резать по предложениям/словам, не ломает слова.
    """
    parts = re.split(r"(\n\n+|(?<=[\.\!\?\:\;])\s+)", text)
    out: List[str] = []
    buf = ""

    def can_add(buf: str, piece: str) -> bool:
        return len(buf) + len(piece) <= max_chars

    for piece in parts:
        if not piece:
            continue
        piece = piece.replace("\n", " ").strip()
        if not piece:
            continue
        if not buf:
            if len(piece) <= max_chars:
                buf = piece
            else:
                # режем по словам
                words = piece.split()
                temp = ""
                for w in words:
                    add = (w if temp == "" else " " + w)
                    if len(temp) + len(add) > max_chars:
                        out.append(temp)
                        temp = w
                    else:
                        temp += add
                if temp:
                    buf = temp
        else:
            add = (" " + piece)
            if can_add(buf, add):
                buf += add
            else:
                out.append(buf)
                if len(piece) <= max_chars:
                    buf = piece
                else:
                    words = piece.split()
                    temp = ""
                    for w in words:
                        add2 = (w if temp == "" else " " + w)
                        if len(temp) + len(add2) > max_chars:
                            out.append(temp)
                            temp = w
                        else:
                            temp += add2
                    buf = temp
    if buf:
        out.append(buf)
    return [s.strip() for s in out if s.strip()]
