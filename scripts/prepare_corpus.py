"""Build a deduplicated text corpus from permitted local UTF-8 sources."""
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
            messages = item.get("messages")
            if isinstance(messages, list):
                dialogue = []
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        role = str(message.get("role", "user")).strip().lower()
                        content = message["content"].strip()
                        if content:
                            label = "GENESIS" if role in {"assistant", "genesis"} else "Пользователь"
                            dialogue.append(f"{label}: {content}")
                if dialogue:
                    yield "\n".join(dialogue)
            else:
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    yield value


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
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/corpus.txt")
    parser.add_argument("--min-chars", type=int, default=20)
    args = parser.parse_args()

    source_dir = Path(args.input)
    output_path = Path(args.output)
    if not source_dir.exists():
        raise SystemExit(f"Input folder not found: {source_dir}. Create it and add text files.")
    seen, chunks = set(), []
    source_count = 0
    for text in collect_text(source_dir):
        source_count += 1
        normalized = text.replace("\r\n", "\n").strip()
        if len(normalized) < args.min_chars:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            chunks.append(normalized)

    if not chunks:
        raise SystemExit("No usable text found. Add permitted UTF-8 text or JSONL dialogue files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Input records/files read: {source_count}")
    print(f"Unique text chunks: {len(chunks)}")
    print(f"Characters: {sum(len(chunk) for chunk in chunks):,}")
    print(f"Output bytes: {output_path.stat().st_size:,}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
