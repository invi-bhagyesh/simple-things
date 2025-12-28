# Interpretability

## Resources:

- [AI Alignment](https://vkrakovna.wordpress.com/ai-safety-resources/#rfp)
- [Neel Nanda Gloassary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
- [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens?tab=readme-ov-file)
- [200 Problems](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability#Overview_of_Sequence) || [excel sheet](https://docs.google.com/spreadsheets/d/1oOdrQ80jDK-aGn-EVdDt3dg65GhmzrvBWzJ6MUZB8n4/edit?gid=0#gid=0)

### Misc:

- [Algoverse Challenge Sheet](https://docs.google.com/document/d/1WKegfb7hPRVAuiRiAsLmxd8AHlOszRUxdauRPePtZAU/edit?tab=t.0)

## Transformer-lens

- Demo
  <a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

## Ideas:

### Project 1: Mech Interp Basics - Replicate Induction Heads

Load GPT-2 Small/XL or Llama-3-8B. Implement basic Logit Lens: project residual stream at each layer to vocab space. Plot how predictions evolve on repeated token sequences (e.g., "A B C D A B C"). Compare raw Logit Lens vs Tuned Lens.

Skills: Model loading, caching activations, basic probing, residual streams.

Time: 1-2 weeks

Resources:

- [TransformerLens docs](https://neelnanda-io.github.io/TransformerLens/)
- [ARENA Chapter 1](https://arena3-chapter1-transformer-interp.streamlit.app/)
- Induction Heads Notebook
  <a target="_blank" href="https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

---

### Project 2: Logit/Tuned Lens Exploration

Take GPT-2 Small or Pythia-410M. Implement both Logit Lens and Tuned Lens on simple prompts. Plot layer-wise top-k predicted tokens and entropy trajectories. Experiment with ambiguous vs clear prompts and sycophantic prompts.

Skills: Internal logit decoding, layer-wise phase changes.

Time: 1-2 weeks

Resources:

- [tuned-lens GitHub](https://github.com/AlignmentResearch/tuned-lens)
- [nnsight tutorials](https://nnsight.net/tutorials/)
- [Logit Lens blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

---

### Project 3: Activation Patching on a Known Circuit

Replicate the Indirect Object Identification (IOI) task. Use TransformerLens for activation patching: identify heads that copy indirect objects, patch activations from clean to corrupted prompts, measure logit recovery. Visualize patching heatmaps.

Skills: Activation patching, path patching, causal attribution.

Time: 2-3 weeks

Resources:

- TransformerLens IOI demo
  <a target="_blank" href="https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- [IOI Paper](https://arxiv.org/abs/2211.00593)
- [Neel Nanda YouTube](https://www.youtube.com/c/neaborman)

---

### Project 4: Evaluate CoT Faithfulness on a Toy Task

Use Llama-3-8B or Mistral-7B. Generate CoT on simple arithmetic/biased questions. Compare final answer when forcing model to follow its own CoT vs zero-shot. Use Tuned Lens to see if truth tokens are predicted mid-model even when CoT diverges.

Skills: CoT prompting, faithfulness checks, linking external trace to internals.

Time: 2 weeks

Resources:

- [Turpin et al. CoT Unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)
- [Faithful-CoT repo](https://github.com/veronica320/Faithful-COT)
- [Lanham et al. Paper](https://arxiv.org/abs/2307.13702)

---

### Project 5: Sycophancy Detection and Probing

Curate 50-100 sycophancy prompts. Generate responses with/without CoT. Use Tuned Lens to inspect layer-wise predictions: does the model internally predict truth early but pivot late? Basic patching from truthful control prompt into sycophantic prompt.

Skills: Conflict dataset creation, spotting internal vs external divergence.

Time: 2-3 weeks

Resources:

- [sycophancy-eval GitHub](https://github.com/meg-tong/sycophancy-eval)
- [Sharma et al. Paper](https://arxiv.org/abs/2310.13548)
- [Anthropic Sycophancy](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations)

---

### Project 6: Prototype a Simple IED Metric

Combine all above on Llama-3-8B with sycophancy dataset. Define truth token via contrastive control runs. Compute per-token KL between mid-layer Tuned Lens logits and final generated token probs. Plot average IED per layer and correlate with CoT length. Try basic patching to locate pivot layers.

Skills: Full pipeline (dataset, generation, probing, metric, patching).

Time: 3-4 weeks

Outcome: Preliminary results for IED proposal.

Resources:

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [Baukit](https://github.com/davidbau/baukit)
- [pyvene](https://github.com/stanfordnlp/pyvene)
