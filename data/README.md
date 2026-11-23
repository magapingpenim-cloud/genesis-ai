# GENESIS training data

Put local training sources in `data/raw/`. This folder is intentionally not populated with a scraped or copyrighted dataset.

Supported formats:
- `.txt` and `.md`: UTF-8 text
- `.jsonl`: one JSON object per line with a `text` or `content` string, or a `messages` array containing `role` and `content`

Use only text you wrote, public-domain material, or datasets whose licenses explicitly permit your intended model-training use. Keep personal information and private conversations out unless you have clear permission from everyone involved.

Prepare a deduplicated corpus from the repository root:

```powershell
New-Item -ItemType Directory -Force data/raw
.\.venv\Scripts\python.exe scripts/prepare_corpus.py
```

The resulting `data/corpus.txt` is local and ignored by Git. The current baseline trains next-byte prediction on this plain text. It does not yet perform instruction fine-tuning, retrieval, or web search.
