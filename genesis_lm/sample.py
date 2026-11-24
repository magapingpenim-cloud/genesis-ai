"""Generate text from a locally trained GENESIS-LM checkpoint."""
import argparse, torch
from model import GenesisLM

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/genesis-lm.pt"); p.add_argument("--prompt",default="Привет!"); p.add_argument("--tokens",type=int,default=300); a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=True); cfg=ckpt["config"]
    model=GenesisLM(**cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    start=torch.tensor([list(a.prompt.encode("utf-8"))],dtype=torch.long,device=device)
    result=model.generate(start,a.tokens)
    print(bytes(result[0].tolist()).decode("utf-8",errors="replace"))
if __name__=="__main__": main()
