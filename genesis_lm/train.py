"""Train GENESIS-LM locally on UTF-8 text files (one or more .txt files)."""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import torch
from genesis_lm.model import GenesisLM
from genesis_lm.tokenizer import GenesisBPETokenizer

def load_corpus(path):
    files = sorted(Path(path).glob("**/*.txt")) if Path(path).is_dir() else [Path(path)]
    chunks = [p.read_text(encoding="utf-8", errors="replace") for p in files if p.is_file()]
    text = "\n\n".join(chunks).strip()
    if len(text.encode("utf-8")) < 1000:
        raise SystemExit("Corpus too small: provide at least 1 KB of UTF-8 text in .txt files.")
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/corpus")
    p.add_argument("--out",default="checkpoints/genesis-lm.pt")
    p.add_argument("--tokenizer",default="tokenizer/genesis-bpe.json")
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--block-size",type=int,default=256)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--layers",type=int,default=4)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--grad-accum",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.block_size < 8 or a.grad_accum < 1: p.error("steps/batch-size must be positive and block-size >= 8")
    torch.manual_seed(a.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(Path(a.data).glob("**/*.txt")) if Path(a.data).is_dir() else [Path(a.data)]
    text = "\\n\\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files if p.is_file()).strip()
    tokenizer_path = Path(a.tokenizer)
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}. Run python -m genesis_lm.tokenizer first.")
    tokenizer = GenesisBPETokenizer.load(tokenizer_path)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split=max(a.block_size+2,int(len(data)*0.9))
    if split >= len(data)-1: split=len(data)-1
    train_data,val_data=data[:split],data[split:]
    if len(train_data)<=a.block_size or len(val_data)<=a.block_size:
        raise SystemExit("Corpus must contain enough bytes for both train and validation splits.")
    def batch(source):
        ix=torch.randint(len(source)-a.block_size-1,(a.batch_size,))
        x=torch.stack([source[i:i+a.block_size] for i in ix])
        y=torch.stack([source[i+1:i+a.block_size+1] for i in ix])
        return x.to(device),y.to(device)
    model=GenesisLM(a.width,a.layers,a.heads,a.block_size,vocab_size=tokenizer.vocab_size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    print(f"device={device} corpus_tokens={len(data)} vocab_size={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}")
    @torch.no_grad()
    def evaluate():
        model.eval(); vals=[]
        for _ in range(20):
            x,y=batch(val_data)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
                _,loss=model(x,y)
            vals.append(loss.item())
        model.train(); return sum(vals)/len(vals)
    started=time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1,a.steps+1):
        x,y=batch(train_data)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
            _,loss=model(x,y)
        scaler.scale(loss / a.grad_accum).backward()
        update_now = (step % a.grad_accum == 0) or (step == a.steps)
        if update_now:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step==1 or step%a.eval_every==0 or step==a.steps:
            val=evaluate()
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val:.4f} val_perplexity={math.exp(min(val,20)):.2f} elapsed={time.time()-started:.1f}s",flush=True)
            Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":model.state_dict(),"config":{"width":a.width,"layers":a.layers,"heads":a.heads,"block_size":a.block_size,"vocab_size":tokenizer.vocab_size},"tokenizer_path":str(tokenizer_path),"step":step,"val_loss":val},a.out)
    print(f"Saved checkpoint: {a.out}")
if __name__=="__main__": main()
