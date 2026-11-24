"""Small local Russian knowledge base with transparent lexical retrieval.

This is retrieval over reviewed facts, not neural generation or model training.
Only sufficiently relevant entries are returned; otherwise the caller can use web search.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "knowledge" / "ru_core.jsonl"
STOP = {
    "что", "это", "такое", "как", "почему", "зачем", "когда", "где", "кто",
    "мне", "тебе", "для", "при", "или", "если", "чем", "какой", "какая",
    "какие", "можно", "нужно", "ли", "а", "и", "в", "на", "по", "из",
    "об", "про", "расскажи", "объясни", "пожалуйста", "простыми", "словами",
}


def terms(text: str) -> list[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))
    return [word for word in words if len(word) > 1 and word not in STOP]


def load_entries():
    if not DATA_PATH.exists():
        return []
    entries = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("answer") and row.get("title"):
            searchable = " ".join([row["title"], *row.get("aliases", [])])
            entries.append({**row, "_terms": set(terms(searchable))})
    return entries


def find_knowledge(query: str, threshold: float = 0.34):
    query_terms = set(terms(query))
    if not query_terms:
        return None
    entries = load_entries()
    if not entries:
        return None
    document_frequency = {}
    for entry in entries:
        for term in entry["_terms"]:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    scored = []
    for entry in entries:
        overlap = query_terms & entry["_terms"]
        if not overlap:
            continue
        weighted = sum(
            math.log(1 + len(entries) / (1 + document_frequency.get(term, 0)))
            for term in overlap
        )
        total = sum(
            math.log(1 + len(entries) / (1 + document_frequency.get(term, 0)))
            for term in query_terms
        )
        score = weighted / max(total, 1e-9)
        scored.append((score, entry))
    if not scored:
        return None
    score, best = max(scored, key=lambda item: item[0])
    if score < threshold:
        return None
    return {"title": best["title"], "answer": best["answer"], "score": round(score, 3)}


def answer_from_knowledge(query: str):
    result = find_knowledge(query)
    if not result:
        return None
    return result["answer"], "GENESIS-local-knowledge"
