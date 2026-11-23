"""Build a deduplicated UTF-8 training corpus from local text and JSONL files.

Supported inputs:
- .txt and .md files
- .jsonl rows with "text" or "content"
- .jsonl rows with "messages": [{"role": "...", "content": "..."}]

Only include data you have permission to use for model training.
"""
import argparse
import hashlib
import json
from pathlib import Path

SUPPORTED_TEXT = {".txt", ".md"}


def extract_jsonl(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON: {path}:{line_number}: {exc}")
            continue
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            value = item.get("text") or item.get("content")
            if isinstance(value, str):
                yield value
            messages = item.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        role = str(message.get("role", ""))
                        content = message["content"].strip()
                        if content:
                            yield f"{role}: {content}" if role else content


def collect_text(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name == "corpus.txt":
            continue
        if any(part.startswith(".") for part in path.relative_to(input_dir).parts):
            continue
        try:
            if path.suffix.lower() in SUPPORTED_TEXT:
                yield path.read_text(encoding="utf-8-sig")
            elif path.suffix.lower() == ".jsonl":
                yield from extract_jsonl(path)
        except (OSError, UnicodeError) as exc:
            print(f"Skipping {path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Prepare a local GENESIS text corpus")
    parser.add_argument("--input", default="data/raw", help="Folder containing permitted source files")
    parser.add_argument("--output", default="data/corpus.txt")
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()

    source_dir = Path(args.input)
    output_path = Path(args.output)
    if not source_dir.exists():
        raise SystemExit(
            f"Input folder not found: {source_dir}. Create it and add .txt, .md, or .jsonl files."
        )

    seen = set()
    chunks = []
    source_count = 0
    for text in collect_text(source_dir):
        source_count += 1
        normalized = " ".join(text.split())
        if len(normalized) < args.min_chars:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        chunks.append(normalized)

    if not chunks:
        raise SystemExit("No usable text found. Add permitted text files or supported JSONL data.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    char_count = sum(len(chunk) for chunk in chunks)
    byte_count = output_path.stat().st_size
    print(f"Input records/files read: {source_count}")
    print(f"Unique text chunks: {len(chunks)}")
    print(f"Characters: {char_count:,}")
    print(f"Output bytes: {byte_count:,}")
    print(f"Saved: {output_path}")
    print("Reminder: a clean corpus is not automatically a question-answer dataset or a capable assistant.")


if __name__ == "__main__":
    main()
