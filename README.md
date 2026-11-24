# GENESIS AI

GENESIS is a self-developed assistant research project. Its own decoder-only Transformer, tokenizer experiments, training pipeline, conversation rules, knowledge base, and web retrieval are developed in this repository. It does not use Ollama, Qwen, ChatGPT, or another external language model.

## Current engine (v0.1)

- A small rule-based conversation layer for greetings and product identity.
- A safe arithmetic evaluator using Python AST (it does not execute arbitrary Python).
- Optional DuckDuckGo HTML search.
- Automatic DuckDuckGo web search for questions the local conversation rules cannot answer.\n- Fetching and extracting readable text from up to three public result pages, with source links.\n- A compact extractive summary based on page text (not a generative LLM).\n- Public-address checks before fetching result pages to reduce SSRF exposure.
- A minimal HTTP backend and browser chat UI.

**Honest limitation:** GENESIS's own small rule engine is not a general LLM. Detailed open-ended answers require further development and training of GENESIS's own language model. Search answers summarize passages extracted from accessible result pages; some sites block automated access, so users should open sources to verify details. The answer composer is extractive, not a reasoning language model. The main research milestone is building and training GENESIS's own language model.

## GENESIS native brain (no Ollama or external LLM)

The native model lives in `genesis_lm/`: a decoder-only Transformer trained to predict the next UTF-8 byte. The training script is tuned to start conservatively on a laptop GPU with 4 GB VRAM (width 128, 4 layers, 4 heads, batch 4, gradient accumulation 4). These are research settings, not a claim of parity with a commercial assistant.

Prepare a training text file from the dialogue JSONL:
```powershell
.venv\\Scripts\\python.exe scripts/dialogues_to_corpus.py
```

Run a small pipeline test on the local GPU:
```powershell
.venv\\Scripts\\python.exe -m genesis_lm.train --data data/corpus --steps 500 --batch-size 2 --block-size 256 --width 128 --layers 4 --heads 4 --grad-accum 4 --out checkpoints/genesis-lm.pt
```

The 80-dialogue starter set is only a smoke test and will be memorized. A useful general assistant requires a much larger, legally usable, diverse corpus, evaluation on held-out questions, and several rounds of pretraining and instruction tuning. Do not interpret a falling training loss alone as intelligence.

The inference interface is `genesis_lm/assistant.py`. It loads only `checkpoints/genesis-lm.pt`, which must be trained locally first. The browser app does not silently call any external model.

## Image generation (local ComfyUI, requested directly in chat)

You can type a normal request such as `Нарисуй футуристический город на Марсе` directly into chat; GENESIS routes it to the image generator automatically. The **✦ Изображение** action is also available. It sends the prompt to a ComfyUI server running on the same machine as the GENESIS Python backend. No paid API key is required, but you must install ComfyUI and download a compatible Stable Diffusion checkpoint yourself.

1. Install ComfyUI using its official instructions: https://docs.comfy.org/get_started
2. Put a Stable Diffusion 1.5 checkpoint file in `ComfyUI/models/checkpoints/` (the default filename expected here is `v1-5-pruned-emaonly.safetensors`).
3. Start ComfyUI with low-VRAM settings, for example: `python main.py --lowvram --force-fp16`.
4. Start GENESIS on the same computer. The image button submits a 512×512, 20-step generation job to `http://127.0.0.1:8188`.

Optional environment variables:
- `GENESIS_COMFYUI_URL` — ComfyUI base URL (default `http://127.0.0.1:8188`).
- `GENESIS_SD_CHECKPOINT` — checkpoint filename shown in the ComfyUI checkpoints folder.

**Hardware note:** an RTX 3050 Laptop GPU with 4 GB VRAM is very limited. SD 1.5 at 512×512 with low-VRAM settings is the realistic starting point; generation may be slow or fail if other GPU-heavy apps are running. SDXL is not the recommended starting point.

**Deployment note:** if GENESIS is hosted on a remote server, ComfyUI must be reachable from that server. A ComfyUI process running on your personal PC at 127.0.0.1 is not automatically reachable by a cloud-hosted GENESIS backend.

## Run locally

Requires Python 3.10+ and no third-party Python packages:

```bash
python app.py
```

Open http://localhost:80, or set `PORT=8080` and open http://localhost:8080.

## API

- `GET /api/health` — engine status.
- `POST /api/chat` — JSON body: `{"message":"...","web_search":true,"history":[]}`.

Web search needs outbound internet access. Local image generation needs ComfyUI and a downloaded checkpoint; it does not use a hosted image API.