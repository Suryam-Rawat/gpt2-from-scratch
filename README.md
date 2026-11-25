# GPT-2 from scratch

A GPT-2 reimplementation in PyTorch — no HuggingFace model class, no nanoGPT import.
Includes training, sampling, FlashAttention (fused SDPA), DDP multi-GPU training,
and a **verification script that loads OpenAI's real 124M GPT-2 weights into this
class** and confirms the architecture is correct.

## Verification (the important part)

`load_gpt2.py` loads HF's `gpt2` checkpoint into this from-scratch model, transposing
the Conv1D weights, and reports:

- **Params: 124.4M** — matches GPT-2 small
- **Loss on natural text: `<TBD>`** — expect ~3.0–3.3 (a wrong architecture gives ~10)
- Coherent generation from `"Hello, I'm a language model,"`

```bash
python load_gpt2.py