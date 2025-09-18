# Book Cleaner + TTS Chunks

Полный пайплайн подготовки книги к озвучке в **Yandex SpeechKit v3** при помощи **GPT‑5 nano (OpenAI API)**: очистка «грязного» TXT → склейка → нарезка на 200‑символьные кусочки → массовая озвучка в MP3.

---

## 📄 Подготовка исходника

Если у вас книга в PDF, можно преобразовать её в текст:

```bash
pdftotext book.pdf book.txt
```

Файл `book.txt` поместите в папку `data/`.

## 🔧 Требования

* Python 3.10+ (рекомендуется 3.12)
* macOS / Linux
* Аккаунт OpenAI (ключ для модели очистки) и ключ Yandex SpeechKit (API Key **или** IAM+Folder)

---

## 📦 Установка

В корне проекта:

```bash
cd ~/Downloads/speechKit_books/v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Настройка окружения

Создайте файл `.env` (либо на основе `.env.example`) и заполните:

```env
# OpenAI (очистка текста)
OPENAI_API_KEY=sk-ваш-ключ
BOOK_PATH=./data/book.txt
OUT_DIR=./out
OPENAI_MODEL=gpt-5-nano
MAX_CONTENT_TOKENS=9500

# Yandex SpeechKit (любой из вариантов)
# Вариант A — API-ключ (проще; FOLDER_ID не нужен)
SPEECHKIT_API_KEY=yc-ваш-api-key

# Вариант B — IAM-токен + Folder ID
# IAM_TOKEN=...  
# FOLDER_ID=...  
```

> В скрипте TTS переменные из `.env` подхватываются автоматически (через `python-dotenv`). Если нужно, можно передать креды флагами `--api-key` или `--iam-token`/`--folder-id` при запуске.

---

## 🧱 Структура проекта

```
book-cleaner-tts/
├── README.md
├── requirements.txt
├── data/
│   └── book.txt
├── out/
│   ├── cleaned_full.txt
│   └── speechkit_chunks/
│       ├── 00001.txt
│       ├── 00002.txt
│       └── ...
├── project_config/
│   └── settings.py
└── scripts/
    ├── clean_and_chunk_book.py
    ├── prepare_jsonl.py
    └── tts_speechkit_v3.py
```

> Папка `config/` была переименована в `project_config/`, чтобы избежать конфликта с внешним пакетом `config` из PyPI. Скрипты запускаются **как модули** (`python -m ...`), чтобы корректно резолвились импорты.

---

## ▶️ Шаг 1. Очистка книги и подготовка чанков

Скрипт читает `data/book.txt`, делит на чанки ≤ 9500 токенов (без разрыва слов), отправляет каждый в GPT‑5 nano с промптом‑«чистильщиком», склеивает ответы и режет на кусочки ≤ 200 символов.

Запуск:

```bash
python -m scripts.clean_and_chunk_book
```

Результат:

* `out/cleaned_full.txt` — цельный очищенный текст.
* `out/speechkit_chunks/00001.txt …` — кусочки по ≤ 200 символов.

> Если видите ошибку импорта настроек — убедитесь, что запускаете из корня проекта и именно через `python -m ...`.

---

## 🧾 (Опционально) Шаг 2. Собрать JSONL

Если нужно JSONL для других пайплайнов:

```bash
python -m scripts.prepare_jsonl
```

Результат: `out/speechkit_chunks.jsonl`

---

## 🔊 Шаг 3. Озвучка через SpeechKit v3 (REST)

Скрипт стримит аудио‑чанки `audioChunk.data` и сохраняет MP3.

Быстрый старт с оптимальными дефолтами (голос **filipp**, скорость **1.1x**, MP3):

```bash
python -m scripts.tts_speechkit_v3 \
  --voice filipp \
  --speed 1.1 \
  --container MP3 \
  --sleep 0.2
```

Пути по умолчанию:

* вход: `./out/speechkit_chunks/`
* выход: `./out/audio/00001.mp3`, `00002.mp3`, …

### Полезные флаги

* `--in-dir ./out/speechkit_chunks` — откуда брать текстовые кусочки.
* `--out-dir ./out/audio` — куда писать аудио.
* `--start 501` — начать с файла `00501.txt` (удобно продолжать после прерывания).
* `--limit 100` — синтезировать N файлов для пробы.
* `--api-key ...` — передать API‑ключ прямо флагом (альтернатива `.env`).
* `--iam-token ... --folder-id ...` — аутентификация через IAM.

### Примеры

Озвучить первые 100 кусочков в MP3:

```bash
python -m scripts.tts_speechkit_v3 --limit 100
```

Догнаться с 501‑го:

```bash
python -m scripts.tts_speechkit_v3 --start 501
```

Сохранить в WAV:

```bash
python -m scripts.tts_speechkit_v3 --container WAV
```

---

## 🛠️ Troubleshooting

* **`[FATAL] Нужен SPEECHKIT_API_KEY или IAM_TOKEN + FOLDER_ID`** — скрипт не нашёл креды. Решения:

  1. положить `SPEECHKIT_API_KEY` в `.env`;
  2. передать `--api-key` флагом;
  3. для IAM — `--iam-token` **и** `--folder-id`.
* **`Unknown role '...' for 'filipp' voice` (HTTP 400)** — указанная роль голосом не поддерживается. Либо не передавайте `--role` (по умолчанию роль отключена), либо используйте голос с поддержкой нужной роли.
* **429/5xx** — временные ограничения/ошибки. Скрипт делает ретраи; при частых 429 увеличьте `--sleep` (например, `0.4`–`1.0`).
* **Прервался процесс** — перезапустите с `--start <N>` (номер следующего файла по списку).

---

## 📑 Примечания

* Разделение на 9500 токенов оставляет запас до лимита 10 000 с учётом системного промпта.
* Чанки по 200 символов не рвут слова (резка по границам предложений/слов).
* Для `gpt-5-nano` опции вроде `temperature` могут быть неподдержаны — мы их не передаём.
* Папка `out/audio` содержит готовые MP3; для дальнейшей склейки можно использовать, например, `ffmpeg` или `pydub` (не входит в текущий проект).

---

## ✅ Краткая шпаргалка по запуску

```bash
# 1) Активировать окружение
source .venv/bin/activate

# 2) Очистка и нарезка
python -m scripts.clean_and_chunk_book

# 3) Озвучка (filipp, 1.1x, MP3)
python -m scripts.tts_speechkit_v3 --voice filipp --speed 1.1 --container MP3
```

---

## 🎼 Склейка MP3 файлов

После озвучки получаем сотни коротких mp3. Их удобно объединить в один файл при помощи **ffmpeg**.

### Вариант A — без перекодирования (быстро)

Если все mp3 имеют одинаковые параметры (обычно так у SpeechKit):

```bash
# из корня проекта
cd out/audio

# создаём список файлов в правильном порядке
ls -1 *.mp3 | sort | awk '{print "file \\"" $0 "\\""}' > list.txt

# склеиваем без перекодирования
ffmpeg -f concat -safe 0 -i list.txt -c copy ../book_full.mp3
```

Проверить результат:

```bash
ffprobe ../book_full.mp3 -hide_banner
```

### Вариант B — с перекодированием (надёжно)

Если `-c copy` даёт ошибку:

```bash
ffmpeg -f concat -safe 0 -i list.txt -c:a libmp3lame -b:a 128k ../book_full.mp3
```

Минус: повторная компрессия (немного хуже качество, дольше), плюс: гарантированно работает.

---
