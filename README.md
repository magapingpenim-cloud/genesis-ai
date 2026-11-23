# GENESIS AI

GENESIS is an early, self-written assistant prototype. It does not call ChatGPT, Ollama, or another hosted language model.

## Current engine (v0.1)

- A small rule-based conversation layer for greetings and product identity.
- A safe arithmetic evaluator using Python AST (it does not execute arbitrary Python).
- Optional DuckDuckGo HTML search.
- Extractive answer composition from search result titles and snippets, with clickable source links.
- A minimal HTTP backend and browser chat UI.

**Honest limitation:** this is not yet a trained neural network and cannot reason freely like a modern LLM. Search answers summarize result snippets; users should open sources to verify details. The next research milestone is building and training our own language model.

## Run locally

Requires Python 3.10+ and no third-party Python packages:

```bash
python app.py
```

Open http://localhost:80, or set `PORT=8080` and open http://localhost:8080.

## API

- `GET /api/health` — engine status.
- `POST /api/chat` — JSON body: `{"message":"...","web_search":true,"history":[]}`.

No API keys or paid model services are required by this prototype. Internet search requires the server to be able to reach DuckDuckGo.