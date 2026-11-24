# GENESIS AI — research blueprint for an own language model

This document is an engineering plan, not a claim that a trained model already exists.

## 1. Current repository reality

The current `app.py` prototype uses hand-written rules, a safe arithmetic evaluator, optional DuckDuckGo search, and extractive composition of search snippets. That is useful application scaffolding, but it is not a neural language model. The next milestone is a separately testable language-model package.

## 2. First trainable model

Start with a small decoder-only Transformer language model. It predicts the next token from preceding tokens:

`P(x_t | x_0, ..., x_{t-1})`

Training uses teacher forcing and cross-entropy loss. For a token sequence `[x0, x1, ... xN]`, the input is `[x0 ... xN-1]` and the target is `[x1 ... xN]`. A causal attention mask prevents the model from seeing future tokens.

The first model is deliberately small. It is a research instrument for validating the tokenizer, data pipeline, loss curve, checkpointing, and sampling—not a competitor to commercial frontier models.

## 3. Components and contracts

1. **Tokenizer** — deterministic encode/decode, reserved special tokens, saved vocabulary, round-trip tests.
2. **Dataset preparation** — plain-text ingestion, normalization, deduplication, document-level train/validation split, provenance and license metadata.
3. **Packed batches** — fixed context length; never allow a training example to leak across the train/validation boundary.
4. **Transformer** — token embeddings, positional information, causal self-attention, feed-forward blocks, final normalization, vocabulary logits.
5. **Objective** — next-token cross-entropy, ignoring padding if padding is introduced.
6. **Trainer** — reproducible seeds, gradient clipping, optimizer, learning-rate schedule, periodic validation, checkpoint/resume.
7. **Sampler** — autoregressive generation with temperature and top-k/top-p controls; stop at EOS and enforce a maximum token budget.
8. **Evaluation** — validation loss/perplexity plus a fixed Russian/English test suite. Do not judge progress from cherry-picked prompts.

## 4. Data rules

- Use only data whose license and permitted use are understood.
- Keep source URL/name, license, collection date, and processing notes alongside each dataset.
- Remove exact duplicates and obvious boilerplate.
- Keep private, personal, credential-bearing, and otherwise sensitive data out of the corpus.
- Do not scrape websites in ways that violate terms or access controls.
- A small clean corpus is preferable to a large opaque corpus for the first experiments.

## 5. Training sequence

### Phase A — pipeline correctness
Create tokenizer and dataset tests. Verify that batches have the expected shapes and that targets are shifted by exactly one token.

### Phase B — tiny overfit test
Train on a tiny synthetic corpus until training loss falls substantially. This verifies that gradients flow and the model can learn the task. It does not demonstrate general intelligence.

### Phase C — small corpus experiment
Train on a licensed, documented corpus. Track training and validation loss separately. If training loss falls while validation loss rises, investigate overfitting and data quality.

### Phase D — generation and evaluation
Generate samples from fixed prompts, compare checkpoints using the same prompts, and record failures. Add tests for deterministic sampling when temperature is zero (or the equivalent greedy mode).

### Phase E — scale only after evidence
Increase context length, width, depth, and dataset size only when the smaller run is stable and measurable. Estimate memory and training time before launching a run.

## 6. Resource and quality reality

Training from random initialization requires substantial text data and compute. A small local model can prove the pipeline and learn narrow patterns, but it will not spontaneously acquire the breadth or reliability of a large pretrained assistant. GPU memory, storage, electricity, and training time remain real costs even when no paid API is used.

Video generation is a separate, much more demanding research track. A language model does not automatically provide a photorealistic video generator. Keep language and video experiments modular rather than implying one prototype already does both.

## 7. Acceptance criteria for the first milestone

- Tokenizer encode/decode tests pass.
- Dataset split is reproducible and has no document overlap.
- Causal mask test proves a token cannot attend to future positions.
- One tiny training run completes and saves a checkpoint.
- Checkpoint can be reloaded and generates a token sequence.
- Validation loss is logged separately from training loss.
- README accurately describes the model as experimental and does not claim frontier-level capability.

## 8. Immediate implementation order

1. Inspect and test the existing application before changing it.
2. Add `src/genesis_lm/` as an isolated package so the current chat app remains stable.
3. Implement tokenizer and dataset utilities with tests.
4. Implement the small causal Transformer and objective.
5. Add a minimal trainer and checkpoint format.
6. Add a command-line sample-generation script.
7. Integrate with the chat interface only after the standalone model runs reliably.
