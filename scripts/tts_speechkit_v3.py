# scripts/tts_speechkit_v3.py

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import requests
from dotenv import load_dotenv
load_dotenv()

API_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"

# Дефолты под задачу: Филипп, 1.1x, MP3
DEFAULT_VOICE = "filipp"
DEFAULT_ROLE = ""              # роль отключена по умолчанию (у filipp role=neutral не поддерживается)
DEFAULT_SPEED = 1.1
DEFAULT_CONTAINER = "MP3"
DEFAULT_RATE_LIMIT_SLEEP = 0.2  # пауза между запросами, сек

# Пути по умолчанию
DEFAULT_IN_DIR = Path("./out/speechkit_chunks")
DEFAULT_OUT_DIR = Path("./out/audio")


# ---------- Авторизация ----------

def build_headers(api_key: Optional[str], iam_token: Optional[str], folder_id: Optional[str]) -> Dict[str, str]:
    """
    v3 поддерживает:
      - Api-Key: Authorization: Api-Key <...> (x-folder-id не нужен)
      - IAM: Authorization: Bearer <...> + x-folder-id: <FOLDER_ID>
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
        return headers
    if iam_token:
        if not folder_id:
            raise RuntimeError("Для IAM-токена укажи FOLDER_ID (заголовок x-folder-id).")
        headers["Authorization"] = f"Bearer {iam_token}"
        headers["x-folder-id"] = folder_id
        return headers
    raise RuntimeError("Нужен SPEECHKIT_API_KEY или IAM_TOKEN + FOLDER_ID.")


# ---------- Формирование тела запроса ----------

def make_request_body(text: str, voice: str, role: str, speed: float, container: str) -> Dict[str, Any]:
    """
    В v3 каждый объект hints содержит ровно одно поле (voice|role|speed...).
    Роль делаем опциональной.
    """
    body: Dict[str, Any] = {
        "text": text,
        "hints": [
            {"voice": voice},
            {"speed": str(speed)},
        ],
        "outputAudioSpec": {
            "containerAudio": {
                "containerAudioType": container  # WAV | OGG_OPUS | MP3
            }
        },
        "loudnessNormalizationType": "LUFS",
    }
    role = (role or "").strip()
    if role:
        body["hints"].insert(1, {"role": role})
    return body


# ---------- Чтение стримингового ответа ----------

def _iter_audio_chunks_ndjson(resp: requests.Response) -> Iterable[bytes]:
    """
    Проходит по NDJSON/построчному JSON и извлекает base64 аудио-чанки.
    Поддерживает варианты:
      - {"audioChunk":{"data":"..."}}
      - {"result":{"audioChunk":{"data":"..."}}}
    """
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        node = obj.get("result", obj)
        chunk = node.get("audioChunk")
        if isinstance(chunk, dict):
            data_b64 = chunk.get("data")
            if data_b64:
                try:
                    yield base64.b64decode(data_b64)
                except Exception:
                    continue


# ---------- Синтез одного кусочка ----------

def synth_one(
    text: str,
    headers: Dict[str, str],
    voice: str,
    role: str,
    speed: float,
    container: str,
    retries: int = 3,
    timeout: int = 90,
) -> bytes:
    body = make_request_body(text, voice=voice, role=role, speed=speed, container=container)

    for attempt in range(1, retries + 1):
        r = requests.post(API_URL, headers=headers, json=body, stream=True, timeout=timeout)

        if r.status_code == 200:
            audio = bytearray()

            # потоковая склейка чанков
            for chunk_bytes in _iter_audio_chunks_ndjson(r):
                audio.extend(chunk_bytes)

            # если не пришло по строкам — пробуем целиком как JSON
            if not audio:
                try:
                    data = r.json()
                    node = data.get("result", data)
                    if "audioChunk" in node and "data" in node["audioChunk"]:
                        audio.extend(base64.b64decode(node["audioChunk"]["data"]))
                except Exception:
                    # на всякий случай, если прислали бинарник
                    if r.content:
                        return bytes(r.content)

            if audio:
                return bytes(audio)

            raise RuntimeError("HTTP 200, но пустой аудио-ответ (нет audioChunk.data).")

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(2 ** (attempt - 1), 8)
            print(f"[WARN] HTTP {r.status_code}, retry {attempt}/{retries} через {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"TTS error HTTP {r.status_code}: {err}")

    raise RuntimeError("Не удалось синтезировать после ретраев.")


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch TTS через Yandex SpeechKit v3 (REST)")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR, help=f"Папка с .txt чанками (по умолчанию: {DEFAULT_IN_DIR})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Куда сохранять аудио (по умолчанию: {DEFAULT_OUT_DIR})")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"Голос (по умолчанию: {DEFAULT_VOICE})")
    p.add_argument("--role", default=DEFAULT_ROLE, help="Опциональная роль (по умолчанию: выключена)")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED, help=f"Скорость (по умолчанию: {DEFAULT_SPEED})")
    p.add_argument("--container", default=DEFAULT_CONTAINER, choices=["WAV", "OGG_OPUS", "MP3"], help=f"Аудио-контейнер (по умолчанию: {DEFAULT_CONTAINER})")
    p.add_argument("--sleep", type=float, default=DEFAULT_RATE_LIMIT_SLEEP, help=f"Пауза между запросами, сек (по умолчанию: {DEFAULT_RATE_LIMIT_SLEEP})")
    p.add_argument("--limit", type=int, default=0, help="Озвучить не больше N файлов (для теста). 0 = все.")
    p.add_argument("--start", type=int, default=1, help="Стартовый индекс файла (1 = 00001.txt).")

    # креды
    p.add_argument("--api-key", default=None, help="SPEECHKIT_API_KEY (если не задан, пробуем IAM токен).")
    p.add_argument("--iam-token", default=None, help="IAM_TOKEN (для Bearer).")
    p.add_argument("--folder-id", default=None, help="FOLDER_ID (обязателен при IAM_TOKEN).")

    args = p.parse_args(argv)

    api_key = args.api_key or os.getenv("SPEECHKIT_API_KEY")
    iam_token = args.iam_token or os.getenv("IAM_TOKEN")
    folder_id = args.folder_id or os.getenv("FOLDER_ID")

    try:
        headers = build_headers(api_key=api_key, iam_token=iam_token, folder_id=folder_id)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"[FATAL] Нет .txt файлов в {in_dir}", file=sys.stderr)
        return 1

    if args.start > 1:
        files = [p for p in files if int(p.stem) >= args.start]
    if args.limit > 0:
        files = files[:args.limit]

    total = len(files)
    print(f"Файлов для синтеза: {total} (voice={args.voice}, speed={args.speed}, container={args.container})")

    ext = ".wav" if args.container == "WAV" else ".ogg" if args.container == "OGG_OPUS" else ".mp3"

    for i, pth in enumerate(files, 1):
        text = pth.read_text(encoding="utf-8").strip()
        target = out_dir / f"{pth.stem}{ext}"

        if target.exists():
            print(f"[{i}/{total}] SKIP {target.name} (уже есть)")
            continue

        try:
            audio = synth_one(
                text=text,
                headers=headers,
                voice=args.voice,
                role=args.role,
                speed=args.speed,
                container=args.container,
            )
            target.write_bytes(audio)
            print(f"[{i}/{total}] OK   → {target.name} ({len(audio)} bytes)")
        except Exception as e:
            print(f"[{i}/{total}] FAIL {pth.name}: {e}", file=sys.stderr)

        time.sleep(args.sleep)

    print(f"Готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
