# GENESIS local knowledge v1

The chat now checks a small reviewed local Russian knowledge corpus before falling back to web search.

- Corpus: `data/knowledge/ru_core.jsonl` (140 reviewed starter entries across school subjects, science, history, computing, safety, sport and practical skills)
- Retrieval: `genesis/knowledge.py` (transparent token-overlap/IDF ranking)
- Chat integration: `app.py`
- Smoke test: `scripts/test_knowledge.py`

## Test locally

```powershell
git pull origin main
.\.venv\Scripts\python.exe scripts/test_knowledge.py
```

Restart the app after pulling:

```powershell
# Stop the previous process with Ctrl+C, then:
.\.venv\Scripts\python.exe app.py
```

Try questions about gravity, GPU, photosynthesis, Baku, and model overfitting.

This milestone is **retrieval, not neural training**: GENESIS finds a relevant reviewed fact and returns its authored explanation. It does not prove that the small character-level model has learned these facts. Unknown or weakly matched questions continue to the web-search fallback.

The corpus is intentionally small and transparent. Expand it with accurate, reviewed entries and cite/source metadata before treating it as a dependable reference. Do not put private personal data into the corpus.
