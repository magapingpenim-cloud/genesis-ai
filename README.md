# GENESIS AI

Минимальный веб-чат GENESIS AI. Сейчас он принимает сообщения, отправляет их на backend, а backend передаёт запрос внешнему движку по адресу из переменной окружения `ENGINE_URL`.

## Запуск локально

Требуется Python 3.10+; внешние Python-пакеты не нужны.

```bash
python app.py
```

Откройте http://localhost:80 (или задайте другой порт через `PORT=8080`).

## Подключение движка

Добавьте переменную окружения:

- `ENGINE_URL` — полный URL POST endpoint вашего движка.
- `ENGINE_API_KEY` — необязательный ключ; отправляется как Bearer token.

Backend отправляет JSON:

```json
{"message":"Текст пользователя","conversation_id":"идентификатор диалога"}
```

Движок должен вернуть JSON с текстовым полем `reply` или `response`.

Проверка сервера: `GET /api/health`.

Пока `ENGINE_URL` не задан, чат честно сообщает, что движок не подключён. Он не имитирует ответы модели.


## AI answers and web search

The chat now sends conversation history to an Ollama-compatible chat model and can optionally search the web with DuckDuckGo HTML results. Set these Amvera environment variables:

- `OLLAMA_API_KEY` — required bearer key for the Ollama cloud API. Never commit it to Git.
- `OLLAMA_MODEL` — optional model name; default `gemma4:31b`.
- `OLLAMA_HOST` — optional API host; default `https://ollama.com`.

In Amvera → Variables, add `OLLAMA_API_KEY`. Web search is toggled in the chat UI. Search results are passed as context to the model and shown as clickable sources. Without the key, the app returns a clear configuration error rather than pretending to answer.
