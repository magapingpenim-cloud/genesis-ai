"""Train GENESIS-LM locally on UTF-8 text files (one or more .txt files)."""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import torch
from model import GenesisLM

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
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--block-size",type=int,default=256)
    p.add_argument("--width",type=int,default=192)
    p.add_argument("--layers",type=int,default=6)
    p.add_argument("--heads",type=int,default=6)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--eval-every",type=int,default=250)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.block_size < 8: p.error("steps/batch-size must be positive and block-size >= 8")
    torch.manual_seed(a.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    data=load_corpus(a.data)
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
    model=GenesisLM(a.width,a.layers,a.heads,a.block_size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    print(f"device={device} corpus_bytes={len(data)} parameters={sum(p.numel() for p in model.parameters()):,}")
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
    for step in range(1,a.steps+1):
        x,y=batch(train_data); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
            _,loss=model(x,y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer); scaler.update()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            val=evaluate()
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val:.4f} val_perplexity={math.exp(min(val,20)):.2f} elapsed={time.time()-started:.1f}s",flush=True)
            Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":model.state_dict(),"config":{"width":a.width,"layers":a.layers,"heads":a.heads,"block_size":a.block_size},"step":step,"val_loss":val},a.out)
    print(f"Saved checkpoint: {a.out}")
if __name__=="__main__": main()
