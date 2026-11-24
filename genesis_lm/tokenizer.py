"""Small self-contained byte-level BPE tokenizer for GENESIS."""
import argparse
import json
from collections import Counter
from pathlib import Path


class GenesisBPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=1000, min_pair_frequency=2):
        tokens = list(text.encode("utf-8"))
        merges = []
        target = max(256, int(vocab_size))
        while 256 + len(merges) < target and len(tokens) > 1:
            counts = Counter(zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_pair_frequency:
                break
            new_id = 256 + len(merges)
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            merges.append(list(pair))
        return cls(merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            best = None
            best_rank = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            new_id = 256 + best_rank
            merged, i = [], 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        pieces = [bytes([i]) if i < 256 else self._expand(i) for i in ids]
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _expand(self, token_id):
        pair = self.merges[token_id - 256]
        return self._expand(pair[0]) + self._expand(pair[1]) if pair[0] >= 256 else bytes([pair[0]]) + (self._expand(pair[1]) if pair[1] >= 256 else bytes([pair[1]]))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"type":"genesis-byte-bpe","merges":self.merges}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "genesis-byte-bpe":
            raise ValueError("Unsupported tokenizer format")
        return cls(data["merges"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/corpus")
    parser.add_argument("--output", default="tokenizer/genesis-bpe.json")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    files = sorted(Path(args.data).glob("**/*.txt")) if Path(args.data).is_dir() else [Path(args.data)]
    text = "\\n\\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files if p.is_file())
    if len(text.encode("utf-8")) < 1000:
        raise SystemExit("Corpus too small: provide at least 1 KB of training text.")
    tokenizer = GenesisBPETokenizer.train(text, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(f"Tokenizer: {args.output}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Merge rules: {len(tokenizer.merges)}")


if __name__ == "__main__":
    main()
