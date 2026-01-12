# 0. Repo and environment checklist (do this first)

Why: make sure all tooling, models, and datasets are available.
What to check / change:

1. repo root contains `data/`, `extraction/`, `steering/`, `evaluation/`, `scripts/`.
2. Python venv or conda active with `torch`, `transformers`, `tqdm`.
3. GPU available and `device_map` works for your model.

Commands:

```bash
# create venv and install basics (example)
python -m venv .venv && source .venv/bin/activate
pip install torch transformers tqdm
# optionally: install git-lfs if pulling model files
```

Expected:

- Python can import torch and transformers.
- Model download will succeed later.

---

# 1. Data: prepare and categorize prompts

Goal: create the four behavioral sets: Accepted Harmless, Refused Harmless, Accepted Harmful, Refused Harmful.
Why: needed for LAT fitting and refusal direction extraction.

Files / places:

- `data/alpaca/` for accepted harmless
- `data/xstest/` for refused harmless
- `data/advbench/` for refused harmful
- `data/jailbreaks/` for accepted harmful

What to do:

1. If you already have these datasets, place them as newline JSON lines or plain text lists in the appropriate folder.
2. If you only have one dataset, split it with labels; otherwise sample from repos you mentioned.

Script to run: `scripts/run_inference.sh` — adapt it to produce label file with model outputs and a binary label `refused: True/False`.

Example `run_inference.sh` skeleton:

```bash
#!/bin/bash
MODEL_PTH=$1   # e.g. NousResearch/Llama-2-7b-chat-hf
INPUT_PTH=$2   # path to file with one prompt per line or jsonl with {"prompt":...}
OUTPUT_PTH=$3  # e.g. outputs/alpaca_results.jsonl
python scripts/run_inference.py --model "$MODEL_PTH" --input "$INPUT_PTH" --output "$OUTPUT_PTH"
```

`run_inference.py` minimal behavior:

- load model and tokenizer
- generate responses (or call greedy)
- mark refused if output starts with refusal token or contains canonical refusal phrases
- write `jsonl` lines: `{"prompt": "..", "generation": "..", "refused": true/false}`

Expected:

- `outputs/alpaca_results.jsonl`, `outputs/xstest_results.jsonl`, etc. with labels.

If you do not have `run_inference.py`, reply and I will provide it. Otherwise continue.

---

# 2. Quick sanity: inspect refusal labelling

Why: ensure refused vs accepted labeling is correct before training LAT or computing means.

What to run:

```python
# tiny-check.py
import json, sys
fn = sys.argv[1]
with open(fn) as f:
    items = [json.loads(l) for l in f]
counts = {"refused":0,"accepted":0}
for it in items:
    if it['refused']: counts['refused'] += 1
    else: counts['accepted'] += 1
print(fn, counts)
# print a few examples
for it in items[:5]: print(it['prompt'][:120], "->", it['refused'])
```

Expected:

- Reasonable ratio of refused vs accepted per dataset. XSTest should have many refused harmless.

---

# 3. Compute mean-diff vectors (baseline)

Goal: create baseline harmfulness and refusal mean difference vectors as sanity baseline.
Why: quick comparison with LAT later.

Where to put code: `extraction/mean_diff.py`

Key steps:

1. Load model, tokenizer
2. For each prompt, run forward and extract hidden states at:
   - `t_inst` layer index L_inst (choose 13 or 14)
   - `t_post-inst` final layer L_post (last or n-1)
3. Save `mean_harm`, `mean_harmless`, `mean_refuse`, `mean_accept`

Minimal code snippet (conceptual):

```python
# extraction/mean_diff.py (concept)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# load prompts labeled (prompt, label_harmful(bool), refused(bool))
# run model with hooks to capture hidden states at the chosen positions
# compute means and save with torch.save(...)
```

Expected output:

- `output/mean_harm.pt`
- `output/mean_refuse.pt`

You will use these as checks, not the final method.

---

# 4. LAT: fit a linear readout for harmfulness at t_inst

Goal: produce `v_harm` and enable per-example score `h(x)`.
Why: LAT converts direction into an operational scalar control variable.

Files:

- `extraction/lat_probe.py`
- Inputs: `outputs/*.jsonl` labeled harmful vs harmless
- Outputs: `output/v_harm.pt`, `output/lat_scores.jsonl`

What to do step by step:

A. Choose layer and token position

- layer: `L_lat = 13` or `14` (middle layer). If model has `num_hidden_layers=N`, choose `N//2`.
- token position: `t_inst` is index of last user instruction token in `input_ids`. Your tokenization function should return the index of that token. If you used prompt templates with separators like `[/INST]`, find its location.

B. Extract dataset-level activations at that layer and token

- For each prompt x produce `h_tinst(x)` with shape `[hidden_dim]` (or `[window, hidden]` if you want a small window).
- Save array of `(h, label_harmful)` for training LAT.

C. Fit LAT

- LAT in simplest form is a linear readout; you can do ordinary least squares or logistic regression for classification. For robustness do ridge regression.
- Implementation: create matrix `H` of shape `[num_examples, hidden_dim]` and labels `y` atomic 0/1 harmful. Fit `w = (H^T H + lambda I)^{-1} H^T y`.
- Normalize `v_harm = w / ||w||`.

Code snippet for fitting:

```python
# extraction/lat_probe.py (fit part)
import numpy as np, torch
# H: numpy array [M, D], y: [M]
lam = 1e-3
D = H.shape[1]
w = np.linalg.solve(H.T @ H + lam * np.eye(D), H.T @ y)
w = w / np.linalg.norm(w)
torch.save(torch.tensor(w, dtype=torch.float32), "output/v_harm.pt")
# compute scores s = H @ w and save them per prompt
```

D. Per example scoring

- compute `s(x) = h(x) dot v_harm`. Save `output/lat_scores.jsonl` with `{"prompt":..., "score": s, "label":...}`

Expected:

- `output/v_harm.pt`
- `output/lat_scores.jsonl` with scores that separate harmful and harmless reasonably well, AUC > 0.8 is a good sign.

Note:

- If `H` is large, use incremental/online ridge or random projection to reduce memory.

---

# 5. Validate LAT and choose threshold tau

Goal: pick a conservative operating threshold `tau` to distinguish harmful vs benign.
Why: threshold determines when to allow refusal logic.

What to compute:

- ROC curve and Precision recall on the validation split.
- Choose `tau` at a desired tradeoff. For the project, start with `tau` = score percentile with 95% true positives on AdvBench while maximizing true negatives on XSTest.

Script: `evaluation/choose_tau.py`

```python
# load lat_scores.jsonl, split val/test, compute ROC and output suggested tau candidates
```

Expected:

- A few candidate taus, e.g. `tau=0.7` or percentile-based like `tau = np.percentile(scores_harmful, 10)` depending on sign.

---

# 6. Compute refusal vector v_refuse at t_post-inst

Goal: extract `v_refuse` as mean difference between refused vs accepted at `t_post-inst`.
Why: we will subtract this vector when we want to suppress refusal.

What to do:

- Using hooks, collect activations at `t_post-inst` for examples labeled refused vs accepted.
- Compute `v_refuse = mean_refused - mean_accepted`
- Normalize and save `output/v_refuse.pt`

Expected:

- `output/v_refuse.pt` saved and inspected.

---

# 7. Implement the piece-wise operator: hooking and runtime steering

Goal: implement inference-time hooks that read `s(x)` at `t_inst` and conditionally modify `t_post-inst` activations.
Why: this is the actual intervention.

Files to create:

- `steering/hooks.py` contains two hooks:
  - `read_harmfulness_hook` registered at layer `L_lat` that computes `s(x)` and stores it in `module` or an external dict keyed by batch index.
  - `apply_piecewise_operator_hook` registered at `t_post-inst` layer that checks stored scores and subtracts `alpha * v_refuse` for those indices where `s(x) <= tau`.

Essential code sketch:

```python
# steering/hooks.py
import torch, functools
scores_store = {}  # keyed by simple batch id or unique token

def read_harmfulness_hook(module, input, output, *, v_harm, idx_token):
    # input or output depending on pre-hook vs forward hook; choose pre-hook for input to block
    hidden = input[0]  # shape [B, T, D] or output[0]
    # extract t_inst activations: hidden[:, idx_token, :]
    h_tinst = hidden[:, idx_token, :].float().detach()   # [B, D]
    s = torch.matmul(h_tinst, v_harm.to(h_tinst.device)) # [B]
    # store scores; we need a stable key, use id(module) + batch number or thread local
    scores_store[id(module)] = s.cpu()  # simple, but better: map by sample order

def apply_piecewise_operator_hook(module, input, output, *, v_refuse, alpha, tau):
    # output[0] is [B, T, D] at t_post-inst
    hidden = output[0]
    # determine indices per batch to modify from previously stored scores. For simplicity, assume single batch.
    scores = scores_store.pop(id(module), None)
    if scores is None:
        return
    scores = scores.to(hidden.device)
    mask = (scores <= tau).float().unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
    # t_post index is -1 if we want final token. Use concrete index if you know it.
    t_post_index = -1
    # create shift vector [D]
    shift = (alpha * v_refuse.to(hidden.device)).view(1,1,-1) # [1,1,D]
    # apply only where mask True
    hidden[:, t_post_index:t_post_index+1, :] = hidden[:, t_post_index:t_post_index+1, :] - mask * shift
    # replace output by modified hidden
    return (hidden, )  # depends on hook signature expecting tuple of outputs
```

How to register hooks during inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
block_modules = model.model.layers  # or model.transformer.h or similar
L_lat = 13
L_post = model.config.num_hidden_layers - 1

v_harm = torch.load("output/v_harm.pt")
v_refuse = torch.load("output/v_refuse.pt")

pre_hook = functools.partial(read_harmfulness_hook, v_harm=v_harm, idx_token=t_inst_index)
post_hook = functools.partial(apply_piecewise_operator_hook, v_refuse=v_refuse, alpha=1.0, tau=chosen_tau)

handles = []
handles.append(block_modules[L_lat].register_forward_hook(pre_hook))
handles.append(block_modules[L_post].register_forward_hook(post_hook))

# run model(...) generation code
# finally remove handles
for h in handles: h.remove()
```

Important practical notes:

- You must coordinate mapping between which score belongs to which sample in batch. For simplicity, start with `batch_size=1`.
- If batch>1 implement per-sample mapping using order or unique ids attached to token embeddings.

Expected:

- Running generation with hooks modifies output when `score <= tau`, leading to fewer refusals for benign prompts.

---

# 8. Basic test: run small evaluation

Goal: sanity check with a small set.
What to run:

- Prepare `tests/sanity_prompts.jsonl` with 8 prompts: 4 XSTest benign-refused examples and 4 AdvBench harmful ones.
- Run inference twice:
  1. baseline model without hooks
  2. model with piece-wise operator enabled

Record:

- for each prompt: baseline response, post-hook response, refused flag.

Expected:

- For benign prompts in XSTest, baseline refused but post-hook accepted or less frequent refusals.
- For harmful prompts in AdvBench, both baseline and post-hook refused.

---

# 9. Sweep tau and alpha

Goal: find best operating point.
What to do:

1. Fix alpha at 1.0 initially
2. Sweep tau over percentiles of validation harmful scores, e.g. [10,20,30,...,90] percentiles
3. For each tau run evaluation across XSTest and AdvBench and compute:
   - Compliance_rate = fraction of XSTest prompts accepted
   - Safety_rate = fraction of AdvBench prompts refused
   - Tradeoff_score = (Compliance_rate + Safety_rate)/2

Commands:

- `scripts/sweep_params.sh` that calls your inference runner with hooks and passes tau, alpha

Expected:

- Pareto frontier of trade-off. Choose tau that maximizes trade-off or meets safety constraint.

---

# 10. Stress testing

Goal: ensure robustness to suffix attacks and context length.
Tests:

- For each prompt, append a set of adversarial suffixes and see how scores and behavior change.
- Roleplay framing and long context injection.

Expected:

- Latent harmfulness score remains stable for harmful prompts, while refusal may fluctuate without our operator.

---

# 11. Logging, debugging, and examples to save

What to log:

- Per sample: prompt, baseline generation, steered generation, score s, mask decision, tau, alpha.
- Save the 50 most informative cases where baseline refused but operator accepted, and where baseline accepted but operator still refused.

Files:

`outputs/diagnostics.jsonl`

---

# 13. Final deliverables and README parts you should produce

- `scripts/run_inference.sh` and `scripts/run_calibration.sh`
- `extraction/lat_probe.py` with `--fit` and `--score` modes
- `steering/hooks.py` implemented and tested
- `evaluation/metrics.py` producing tradeoff report
- Notebook `notebooks/analysis.ipynb` with plots: score histogram, ROC, tradeoff sweep plot, example diffs.

---

# Quick checklist to start coding now

1. Prepare labeled inference outputs for the four datasets. If missing, create `scripts/run_inference.py`. Run for `batch_size=1`.
2. Implement `extraction/lat_probe.py` to extract hidden states at `L_lat` and fit ridge regression, save `v_harm.pt`.
3. Implement `extraction/refusal_vector.py` to compute `v_refuse.pt` at `t_post-inst`.
4. Implement `steering/hooks.py` and test with `batch_size=1` for a few prompts.
5. Run `evaluation/choose_tau.py` to pick tau, then sweep and report.
