"""Inference for GENESIS's own byte-level Transformer checkpoint."""
from pathlib import Path
import torch
from genesis_lm.model import GenesisLM
from genesis_lm.tokenizer import GenesisBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "genesis-lm.pt"


class GenesisBrain:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.path = Path(checkpoint)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Native GENESIS checkpoint not found: {self.path}. Train genesis_lm first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(self.path, map_location=self.device, weights_only=True)
        config = saved["config"]
        self.model = GenesisLM(
            width=config["width"], layers=config["layers"], heads=config["heads"],
            block_size=config["block_size"], dropout=0.0, vocab_size=config.get("vocab_size", 256)
        ).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.step = int(saved.get("step", 0))
        tokenizer_path = saved.get("tokenizer_path", "tokenizer/genesis-bpe.json")
        self.tokenizer = GenesisBPETokenizer.load(ROOT / tokenizer_path)

    @torch.inference_mode()
    def answer(self, message, history=None, max_new_tokens=420):
        turns = []
        for item in (history or [])[-8:]:
            if not isinstance(item, dict):
                continue
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                turns.append(("Пользователь" if role == "user" else "GENESIS", content[:1200]))
        if not turns or turns[-1] != ("Пользователь", message):
            turns.append(("Пользователь", message[:2000]))
        prompt = (
            "Ты GENESIS, русскоязычный ИИ-помощник. Отвечай понятно и по делу. "
            "Не выдумывай факты.\n\n"
            + "\n".join(f"{role}: {content}" for role, content in turns)
            + "\nGENESIS:"
        )
        encoded = self.tokenizer.encode(prompt)
        encoded = encoded[-self.model.block_size:]
        tokens = torch.tensor(encoded, dtype=torch.long, device=self.device)[None, :]
        generated = self.model.generate(tokens, max_new_tokens=max_new_tokens,
                                        temperature=0.75, top_k=40)
        result = self.tokenizer.decode(generated[0, tokens.shape[1]:].tolist())
        for marker in ("\nПользователь:", "\nТы:", "\n###"):
            if marker in result:
                result = result.split(marker, 1)[0]
        return result.strip()
