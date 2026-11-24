# GENESIS AI

GENESIS is an early, self-written assistant prototype. It does not call ChatGPT, Ollama, or another hosted language model.

## Current engine (v0.1)

- A small rule-based conversation layer for greetings and product identity.
- A safe arithmetic evaluator using Python AST (it does not execute arbitrary Python).
- Optional DuckDuckGo HTML search.
- Automatic DuckDuckGo web search for questions the local conversation rules cannot answer.\n- Fetching and extracting readable text from up to three public result pages, with source links.\n- A compact extractive summary based on page text (not a generative LLM).\n- Public-address checks before fetching result pages to reduce SSRF exposure.
- A minimal HTTP backend and browser chat UI.

**Honest limitation:** this is not yet a trained neural network and cannot reason freely like a modern LLM. Search answers summarize passages extracted from accessible result pages; some sites block automated access, so users should open sources to verify details. The answer composer is extractive, not a reasoning language model. The next research milestone is building and training our own language model.

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