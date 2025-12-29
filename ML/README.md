# ML Practice Template

## Progress Tracker

| #   | Topic                 | File                                   | Status | Last Practice |
| --- | --------------------- | -------------------------------------- | ------ | ------------- |
| 1   | 2-layer MLP (FWD+BWD) | [01_mlp.py](01_mlp.py)                 | -      | -             |
| 2   | Multi-head Attention  | [02_attention.py](02_attention.py)     | -      | -             |
| 3   | RoPE                  | [03_rope.py](03_rope.py)               | -      | -             |
| 4   | KV Cache              | [04_kv_cache.py](04_kv_cache.py)       | -      | -             |
| 5   | Parallelism (TP/DP)   | [05_parallelism.py](05_parallelism.py) | -      | -             |

**Legend:** `-` Not started | `WIP` In progress | `DONE` Confident

---

## Implementation Methods

- Get exercise + test cases + scaffold from chatbot
- Implement using only **NumPy + stdlib**
- First attempts: small nudges allowed
- Later: no references, grind it out
- Validate with chatbot feedback

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

---

## References

- [Two Pug](https://twopug.com/interview-prep-ml-grind/) (pending)
- [cheatsheet.md](cheatsheet.md) - Shapes & FLOPs quick reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Megatron-LM (Tensor Parallelism)](https://arxiv.org/abs/1909.08053)
