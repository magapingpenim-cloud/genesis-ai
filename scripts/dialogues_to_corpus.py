"""Prepare a plain-text conversation corpus from GENESIS JSONL dialogue data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/starter_dialogues.jsonl")
    parser.add_argument("--output", default="data/corpus/dialogues.txt")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"Dialogue file not found: {source}")
    rows = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON at line {line_no}")
            continue
        messages = item.get("messages", []) if isinstance(item, dict) else []
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                label = "Пользователь" if role == "user" else "GENESIS"
                parts.append(f"{label}: {content.strip()}")
        if parts:
            rows.append("\\n".join(parts))
    if not rows:
        raise SystemExit("No usable conversations found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\\n\\n".join(rows) + "\\n", encoding="utf-8")
    print(f"Conversations: {len(rows)}")
    print(f"Corpus: {output} ({output.stat().st_size:,} bytes)")
    print("Note: this is only a pipeline smoke-test corpus, not enough to create a capable general assistant.")


if __name__ == "__main__":
    main()
