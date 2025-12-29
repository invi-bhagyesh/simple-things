# ML Practice Template

## Reading List

> **Status:** `-` Not read | `WIP` In progress | `DONE` Completed

| Status | Resource                                                                   |
| ------ | -------------------------------------------------------------------------- |
| -      | [Two Pug - ML Interview Prep](https://twopug.com/interview-prep-ml-grind/) |
| -      | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)              |
| -      | [RoFormer: RoPE](https://arxiv.org/abs/2104.09864)                         |
| -      | [GQA Paper](https://arxiv.org/abs/2305.13245)                              |
| -      | [Speculative Decoding](https://arxiv.org/abs/2211.17192)                   |
| -      | [Megatron-LM (Tensor Parallelism)](https://arxiv.org/abs/1909.08053)       |
| -      | [cheatsheet.md](cheatsheet.md)                                             |

---

## Status

| #   | Topic                 | File                                   | Status | Last Practice |
| --- | --------------------- | -------------------------------------- | ------ | ------------- |
| 1   | 2-layer MLP (FWD+BWD) | [01_mlp.py](01_mlp.py)                 | -      | -             |
| 2   | Multi-head Attention  | [02_attention.py](02_attention.py)     | -      | -             |
| 3   | RoPE                  | [03_rope.py](03_rope.py)               | -      | -             |
| 4   | KV Cache              | [04_kv_cache.py](04_kv_cache.py)       | -      | -             |
| 5   | Parallelism (TP/DP)   | [05_parallelism.py](05_parallelism.py) | -      | -             |

---

## Implementation Methods

1. Get exercise + test cases + scaffold from chatbot
2. Implement using only **NumPy + stdlib**
3. First attempts: small nudges allowed
4. Later: no references, grind it out
5. Validate with chatbot feedback

---

## Topics

| Topic                | Understand                       |
| -------------------- | -------------------------------- |
| GQA                  | Grouped Query Attention          |
| MLA                  | Multi-Latent Attention           |
| RoPE                 | Rotary position embeddings       |
| KV Cache             | Prefill vs decode, memory layout |
| Speculative Decoding | Draft + verify, speedup analysis |
| Continuous Batching  | Dynamic batch scheduling         |
