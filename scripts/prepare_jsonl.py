# scripts/prepare_jsonl.py

import argparse
import json
import sys
from pathlib import Path

# 1) Настройки из .env (если есть)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 2) Настройки из config/settings.py (если есть)
DEFAULT_OUT_DIR = "./out"
DEFAULT_CHUNK_DIRNAME = "speechkit_chunks"
DEFAULT_JSONL_NAME = "speechkit_chunks.jsonl"
DEFAULT_MAX_CHARS = 200

try:
    from project_config.settings import OUT_DIR as CFG_OUT_DIR  # type: ignore
except Exception:
    CFG_OUT_DIR = DEFAULT_OUT_DIR

try:
    from project_config.settings import SPEECHKIT_CHUNK_SIZE as CFG_MAX_CHARS  # type: ignore
except Exception:
    CFG_MAX_CHARS = DEFAULT_MAX_CHARS


def natural_key(s: str):
    """Ключ для «человеческой» сортировки имен файлов: 00001.txt < 00010.txt < 000100.txt."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_chunks(in_dir: Path) -> list[tuple[str, str]]:
    """
    Читает все .txt в директории, возвращает список (id, text).
    id — имя файла без расширения.
    """
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Директория с чанками не найдена: {in_dir}")

    files = sorted((p for p in in_dir.glob("*.txt") if p.is_file()),
                   key=lambda p: natural_key(p.name))

    if not files:
        raise FileNotFoundError(f"В {in_dir} нет .txt файлов с чанками")

    items: list[tuple[str, str]] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="strict").strip()
        items.append((p.stem, text))
    return items


def validate_lengths(items: list[tuple[str, str]], max_chars: int) -> list[str]:
    """
    Проверяет, что каждый кусок ≤ max_chars. Возвращает список предупреждений.
    """
    warnings: list[str] = []
    for stem, text in items:
        if len(text) > max_chars:
            warnings.append(
                f"[WARN] {stem}.txt: длина {len(text)} символов > {max_chars}. "
                f"Рекомендуется перепорезать."
            )
    return warnings


def write_jsonl(items: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for stem, text in items:
            line = {"id": stem, "text": text}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Собирает чанки TTS (.txt) в один JSONL для Yandex SpeechKit."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path(CFG_OUT_DIR) / DEFAULT_CHUNK_DIRNAME,
        help=f"Папка с .txt чанками (по умолчанию: {Path(CFG_OUT_DIR) / DEFAULT_CHUNK_DIRNAME})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(CFG_OUT_DIR) / DEFAULT_JSONL_NAME,
        help=f"Куда сохранить JSONL (по умолчанию: {Path(CFG_OUT_DIR) / DEFAULT_JSONL_NAME})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=CFG_MAX_CHARS,
        help=f"Максимальная длина кусочка в символах (для валидации; по умолчанию: {CFG_MAX_CHARS})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Не проверять длину кусочков.",
    )

    args = parser.parse_args(argv)

    items = read_chunks(args.in_dir)

    if not args.no_validate:
        warns = validate_lengths(items, args.max_chars)
        for w in warns:
            print(w, file=sys.stderr)
        if warns:
            print(f"\nИтого предупреждений: {len(warns)}", file=sys.stderr)

    write_jsonl(items, args.out)

    print(f"OK: {args.out} (записано {len(items)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
